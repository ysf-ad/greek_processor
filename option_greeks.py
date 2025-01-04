import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Optional

@dataclass
class OptionGreeks:
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float

def calculate_bsm_iv(S: float, K: float, r: float, T: float, price: float, is_call: bool, 
                     tolerance: float = 0.0001, max_iter: int = 100) -> Optional[float]:
    """
    Calculate implied volatility using Newton-Raphson method
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        T: Time to expiration in years
        price: Option price
        is_call: True for call, False for put
        tolerance: Convergence tolerance
        max_iter: Maximum iterations
    """
    # Initial guess for IV
    sigma = 0.3
    
    for i in range(max_iter):
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if is_call:
            price_calc = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            price_calc = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
        diff = price - price_calc
        
        if abs(diff) < tolerance:
            return sigma
        
        # Vega calculation
        vega = S*np.sqrt(T)*norm.pdf(d1)
        
        if vega == 0:
            return None
            
        # Update sigma
        sigma = sigma + diff/vega
        
        if sigma <= 0:
            return None
    
    return None

def calculate_greeks(S: float, K: float, r: float, T: float, sigma: float, is_call: bool) -> OptionGreeks:
    """Calculate all option Greeks using BSM"""
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    # Delta
    delta = norm.cdf(d1) if is_call else -norm.cdf(-d1)
    
    # Gamma
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    
    # Theta
    theta = (-S*sigma*norm.pdf(d1))/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2 if is_call else -d2)
    
    # Vega
    vega = S*np.sqrt(T)*norm.pdf(d1)
    
    return OptionGreeks(sigma, delta, gamma, theta, vega) 