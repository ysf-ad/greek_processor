import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, leastsq
from dataclasses import dataclass
from typing import Optional, List, Tuple
import scipy.optimize as opt

@dataclass
class SVIParams:
    a: float      # Overall level
    b: float      # Asymmetry
    rho: float    # Correlation/skew
    m: float      # ATM shift
    sigma: float  # ATM volatility

def straight_svi(x: float, m1: float, m2: float, q1: float, q2: float, c: float) -> float:
    """Hyperbola asymptotes parametrisation of SVI"""
    return ((m1 + m2)*x + q1 + q2 + 
            np.sqrt(((m1 + m2)*x + q1 + q2)**2 - 
                   4*(m1*m2*x**2 + (m1*q2 + m2*q1)*x + q1*q2 - c)))/2

def straight_svi_prime(x: float, m1: float, m2: float, q1: float, q2: float, c: float) -> float:
    """First derivative of straight SVI"""
    H = np.sqrt(((m1 + m2)*x + q1 + q2)**2 - 4*(m1*m2*x**2 + (m1*q2 + m2*q1)*x + q1*q2 - c))
    return ((m1 + m2) + ((m1 + m2)*((m1 + m2)*x + q1 + q2) - 
            4*m1*m2*x - 2*(m1*q2 + m2*q1))/H)/2

def straight_svi_double_prime(x: float, m1: float, m2: float, q1: float, q2: float, c: float) -> float:
    """Second derivative of straight SVI"""
    H = np.sqrt(((m1 + m2)*x + q1 + q2)**2 - 4*(m1*m2*x**2 + (m1*q2 + m2*q1)*x + q1*q2 - c))
    A = (2*(m1 + m2)**2 - 8*m1*m2)/H
    B = (2*(m1 + m2)*((m1 + m2)*x + q1 + q2) - 8*m1*m2*x - 4*(m1*q2 + m2*q1))**2/H**3/2
    return (A - B)/4

def butterfly_arbitrage(chi: List[float], k: float) -> float:
    """Calculate butterfly arbitrage at a point"""
    m1, m2, q1, q2, c = chi
    w = straight_svi(k, m1, m2, q1, q2, c)
    wp = straight_svi_prime(k, m1, m2, q1, q2, c)
    wpp = straight_svi_double_prime(k, m1, m2, q1, q2, c)
    g = (1 - k*wp/(2*w))**2 - wp**2/4*(1/w + 1/4) + wpp/2
    return -min(0, g)

def calendar_arbitrage(chi1: List[float], chi2: List[float], k: float) -> float:
    """Calculate calendar arbitrage between two slices at a point"""
    w1 = straight_svi(k, *chi1)
    w2 = straight_svi(k, *chi2)
    return max(0, w2 - w1)

def fit_svi_slice(strikes: List[float], ivs: List[float], spot: float, 
                  prev_params: Optional[List[float]] = None) -> Optional[SVIParams]:
    """
    Fit SVI parameters with arbitrage penalties
    
    Args:
        strikes: List of option strikes
        ivs: List of implied volatilities
        spot: Current spot price
        prev_params: Parameters from previous slice for calendar spread
    """
    if len(strikes) < 5:
        return None
        
    # Convert to log-moneyness and variance
    x = np.array([np.log(K/spot) for K in strikes])
    v = np.array([iv*iv for iv in ivs])
    
    # Split data into 5 intervals for initial guess
    kmin, kmax = min(x), max(x)
    intervals = np.linspace(kmin, kmax, 6)
    xm = []
    vm = []
    
    for i in range(5):
        mask = (x >= intervals[i]) & (x < intervals[i+1])
        if np.any(mask):
            xm.append(np.mean(x[mask]))
            vm.append(np.mean(v[mask]))
    
    xm = np.array(xm)
    vm = np.array(vm)
    
    def residuals(chi):
        m1, m2, q1, q2, c = chi
        # Basic fit error
        w = np.array([straight_svi(k, m1, m2, q1, q2, c) for k in x])
        fit_error = v - w
        
        # Butterfly arbitrage penalty
        but_penalty = np.sum([butterfly_arbitrage(chi, k) for k in x])
        
        # Calendar arbitrage penalty if we have previous parameters
        cal_penalty = 0
        if prev_params is not None:
            cal_penalty = np.sum([calendar_arbitrage(chi, prev_params, k) for k in x])
        
        # Combine errors with penalties
        total_error = fit_error + 10*but_penalty + 100*cal_penalty
        return total_error
    
    # Initial guess
    init_guess = [
        -0.2,  # m1
        0.2,   # m2
        0.1,   # q1
        0.1,   # q2
        0.01   # c
    ]
    
    # Bounds
    bounds = [
        (-0.5, 0.5),    # m1
        (-0.5, 0.5),    # m2
        (0, 0.5),       # q1
        (0, 0.5),       # q2
        (0.001, 0.1)    # c
    ]
    
    try:
        result = minimize(lambda x: np.sum(residuals(x)**2), 
                         init_guess,
                         method='L-BFGS-B',
                         bounds=bounds)
        
        if result.success:
            m1, m2, q1, q2, c = result.x
            # Convert to raw SVI parameters
            a = (m1*q2 - m2*q1)/(m1 - m2)
            b = abs(m1 - m2)/2
            rho = (m1 + m2)/abs(m1 - m2)
            m = -(q1 - q2)/(m1 - m2)
            sigma = np.sqrt(4*c)/abs(m1 - m2)
            
            return SVIParams(a, b, rho, m, sigma)
            
    except Exception as e:
        print(f"SVI fit failed: {str(e)}")
    
    return None

def get_svi_slice(trades: List[Tuple[float, float, float]], spot: float) -> Optional[SVIParams]:
    """Get SVI parameters for a slice of trades"""
    if len(trades) < 5:
        return None
    
    # Sort by strike and take mean IV for duplicates
    unique_trades = {}
    for strike, iv, _ in trades:
        if strike not in unique_trades:
            unique_trades[strike] = [iv]
        else:
            unique_trades[strike].append(iv)
    
    strikes = []
    ivs = []
    for strike, iv_list in sorted(unique_trades.items()):
        strikes.append(strike)
        ivs.append(np.mean(iv_list))
    
    return fit_svi_slice(strikes, ivs, spot) 

def get_svi_iv(k: float, params: SVIParams) -> float:
    """
    Get implied volatility from SVI parameters for a given log-moneyness
    
    Args:
        k: Log-moneyness (log(K/S))
        params: SVI parameters
    """
    # Convert raw SVI parameters to straight SVI parameters
    b = params.b
    if abs(b) < 1e-8:
        return np.sqrt(params.a)
        
    rho = max(min(params.rho, 0.99), -0.99)  # Ensure |rho| < 1
    m1 = b * (rho + 1)
    m2 = b * (rho - 1)
    
    # Calculate q1, q2
    q1 = params.a + b * (rho * (params.m) + np.sqrt((params.m)**2 + params.sigma**2))
    q2 = params.a + b * (rho * (params.m) - np.sqrt((params.m)**2 + params.sigma**2))
    
    # Calculate c
    c = (q1 * q2 - params.a**2) / (4 * b**2)
    
    try:
        # Calculate total variance using straight SVI
        w = straight_svi(k, m1, m2, q1, q2, c)
        return np.sqrt(max(0, w))  # Ensure non-negative variance
    except:
        return params.a  # Fallback to constant volatility if calculation fails 