import market_data
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility as iv_bsm
from matplotlib.widgets import Slider
from scipy.optimize import minimize
from scipy import optimize

@dataclass
class Trade:
    time: float
    ms_of_day: int
    price: float
    size: int
    right: str
    strike: float
    spot_price: float
    bid: float = None
    ask: float = None
    is_buyer: bool = None  # Add aggressor flag
    iv: float = None

def calculate_iv(trade: Trade) -> float:
    try:
        T = max((16 - trade.time) / 24 / 365, 1/365/24)
        r = 0.05
        q = 0.015
        
        iv = iv_bsm(
            price=trade.price,
            S=trade.spot_price,
            K=trade.strike,
            t=T,
            r=r,
            q=q,
            flag=trade.right.lower()
        )
        
        return iv if 0 < iv < 5 else None
    except:
        return None

def evaluate_series(x, coeffs):
    """Evaluate Taylor series at x given coefficients"""
    result = coeffs[0]  # constant term
    x_term = x.copy()
    for c in coeffs[1:]:
        result += c * x_term
        x_term = x_term * x / (len(coeffs))  # normalize to prevent overflow
    return result

def fit_taylor_series(x, y, trades, max_degree=12):
    """
    Fit a Taylor series to the midpoint of trades at each strike.
    Allow high degree fitting to get exact match to midpoints.
    """
    # Group trades by strike
    strike_data = {}
    for i, (xi, yi) in enumerate(zip(x, y)):
        trade = trades[i]
        strike = trade.strike
        if strike not in strike_data:
            strike_data[strike] = []
        strike_data[strike].append(yi)
    
    # Convert to relative strikes and get midpoints
    latest_spot = trades[-1].spot_price
    final_x = []
    final_y = []
    
    for strike in sorted(strike_data.keys()):
        ivs = strike_data[strike]
        relative_strike = (strike/latest_spot - 1) * 100
        # Use median IV for each strike (more robust than mean)
        final_x.append(relative_strike)
        final_y.append(np.median(ivs))
    
    # Convert to numpy arrays
    final_x = np.array(final_x)
    final_y = np.array(final_y)
    
    # Try increasing degrees until we get a good fit
    best_coeffs = None
    best_loss = float('inf')
    
    for degree in range(2, max_degree + 1):
        if len(final_x) <= degree:
            break
            
        # Fit polynomial of current degree
        coeffs = np.polyfit(final_x, final_y, degree)
        
        # Calculate loss
        y_pred = np.polyval(coeffs, final_x)
        loss = np.mean((final_y - y_pred) ** 2)
        
        print(f"Degree {degree} loss: {loss:.8f}")
        
        # Keep if it's better
        if loss < best_loss:
            best_coeffs = coeffs
            best_loss = loss
            
            # If fit is very good, we can stop
            if loss < 1e-10:
                print(f"Found excellent fit at degree {degree}")
                break
    
    print(f"Final Taylor series degree: {len(best_coeffs)-1}")
    return best_coeffs

def fit_polynomial(x, y, degree=4):
    """Simple polynomial fit to the midpoints"""
    return np.polyfit(x, y, degree)

def update(val):
    try:
        current_time = time_slider.val
        zoom = zoom_slider.val
        x_pan = x_pan_slider.val
        y_pan = y_pan_slider.val
        
        # Clear the plot
        ax.clear()
        
        # Find current index
        current_idx = next(i for i, (ms, t) in enumerate(all_trades) if t.time >= current_time)
        start_idx = max(0, current_idx - 100)
        end_idx = min(len(all_trades), current_idx + 100)
        window_trades = [t for _, t in all_trades[start_idx:end_idx]]
        
        if window_trades:
            latest_spot = window_trades[-1].spot_price
            print(f"Processing {len(window_trades)} trades")
            
            # Calculate all relative strikes and IVs first
            relative_strikes = np.array([(t.strike/latest_spot - 1) * 100 for t in window_trades])
            ivs = np.array([t.iv for t in window_trades])
            print(f"Got {len(ivs)} trade IVs")
            
            # Group trades by strike and calculate midpoints
            strike_data = {}
            for t, rel_strike, iv in zip(window_trades, relative_strikes, ivs):
                if rel_strike not in strike_data:
                    strike_data[rel_strike] = []
                strike_data[rel_strike].append(iv)
            
            # Calculate midpoints for each strike
            mid_strikes = []
            mid_ivs = []
            for strike in sorted(strike_data.keys()):
                strike_ivs = strike_data[strike]
                mid_strikes.append(strike)
                mid_ivs.append(np.median(strike_ivs))
            
            mid_strikes = np.array(mid_strikes)
            mid_ivs = np.array(mid_ivs)
            print(f"Calculated {len(mid_ivs)} strike midpoints")
            
            # Separate buyer and seller trades for plotting
            buyer_mask = np.array([t.is_buyer for t in window_trades])
            seller_mask = ~buyer_mask
            
            # Plot trades with smaller dots
            ax.scatter(relative_strikes[buyer_mask], ivs[buyer_mask] * 100, 
                      s=1, color='green', alpha=0.3, label='Buyer Initiated')
            ax.scatter(relative_strikes[seller_mask], ivs[seller_mask] * 100, 
                      s=1, color='red', alpha=0.3, label='Seller Initiated')
            
            # Plot strike midpoints with larger dots
            ax.scatter(mid_strikes, mid_ivs * 100, color='blue', 
                      s=20, alpha=0.7, label='Strike Midpoint')
            
            # Fit polynomial to midpoints
            coeffs = fit_polynomial(mid_strikes, mid_ivs)
            
            # Create smooth curve for polynomial
            x_smooth = np.linspace(min(relative_strikes), max(relative_strikes), 100)
            y_smooth = np.polyval(coeffs, x_smooth)
            
            # Plot polynomial curve
            ax.plot(x_smooth, y_smooth * 100, 'b-', linewidth=1.5, alpha=0.8,
                   label='Polynomial Fit')
            
            # Vertical line at ATM
            ax.axvline(x=0, color='blue', linestyle='--', alpha=0.3, 
                      label=f'ATM (Spot: {latest_spot:.0f})')
            
            # Calculate view bounds
            x_center = 0
            y_center = np.mean(ivs) * 100
            
            x_range = 15
            y_range = 20
            
            x_min = x_center - (x_range / zoom) + x_pan
            x_max = x_center + (x_range / zoom) + x_pan
            y_min = y_center - (y_range / zoom) + y_pan
            y_max = y_center + (y_range / zoom) + y_pan
            
            ax.set_xlabel('% Away from Spot')
            ax.set_ylabel('Implied Volatility (%)')
            ax.set_title(f'Option Trades ({current_time:.2f} hours)')
            ax.grid(True, alpha=0.2)
            ax.legend(loc='upper right', fontsize=8)
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(max(0, y_min), y_max)
            
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_visible(True)
        
        fig.canvas.draw_idle()
            
    except Exception as e:
        print(f"Error in update: {str(e)}")
        import traceback
        traceback.print_exc()

def plot_trades():
    global time_slider, zoom_slider, x_pan_slider, y_pan_slider, ax, all_trades, fig
    
    # Initialize and load data
    md = market_data.MarketData()
    target_date = 20250103  # Use integer format
    root = "SPXW"
    
    print("Loading spot price data...")
    md.load_spot_prices(root, str(target_date))
    if not md.spot_price_data:
        print("Failed to load spot price data")
        return
    
    print("Loading trade and quote data...")
    trade_data = market_data.MarketData.get_day_trade_quotes(root, str(target_date))
    if not trade_data:
        print("No trades found")
        return
    
    # Get indices once
    header_format = trade_data["header"]["format"]
    TIME_IDX = header_format.index("ms_of_day")
    PRICE_IDX = header_format.index("price")
    SIZE_IDX = header_format.index("size")
    BID_IDX = header_format.index("bid")
    ASK_IDX = header_format.index("ask")
    
    # Pre-process spot price data
    spot_times = sorted(md.spot_price_data.keys())
    
    # Collect ALL trades with their timestamps
    all_trades = []
    
    print(f"Number of responses: {len(trade_data['response'])}")
    
    for response in trade_data["response"]:
        expiry = response["contract"]["expiration"]
        if str(expiry) != str(target_date):
            continue
            
        strike = response["contract"]["strike"] / 1000
        
        for tick in response["ticks"]:
            ms_of_day = tick[TIME_IDX]
            rounded_ms = round(ms_of_day / 500) * 500
            spot_price = md.spot_price_data.get(rounded_ms, md.spot_price_data[spot_times[0]])
            
            # Pre-classify the trade
            mid_price = (tick[BID_IDX] + tick[ASK_IDX]) / 2
            is_buyer = tick[PRICE_IDX] >= mid_price
            
            trade = Trade(
                time=ms_of_day / (1000 * 3600),
                ms_of_day=ms_of_day,
                price=tick[PRICE_IDX],
                size=tick[SIZE_IDX],
                right=response["contract"]["right"],
                strike=strike,
                spot_price=spot_price,
                bid=tick[BID_IDX],
                ask=tick[ASK_IDX],
                is_buyer=is_buyer  # Store classification
            )
            trade.iv = calculate_iv(trade)
            if trade.iv is not None:
                all_trades.append((ms_of_day, trade))
    
    # Sort by time
    all_trades.sort(key=lambda x: x[0])
    print(f"Collected {len(all_trades)} valid trades")
    
    if not all_trades:
        print("No valid trades to plot")
        return
    
    print("Starting to create plot...")
    
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 8))
        plt.subplots_adjust(bottom=0.3)  # More space for sliders
        
        # Calculate time range
        min_time = all_trades[0][1].time
        max_time = all_trades[-1][1].time
        initial_time = min_time
        
        # Create sliders
        time_slider_ax = plt.axes([0.1, 0.15, 0.65, 0.03])
        zoom_slider_ax = plt.axes([0.1, 0.10, 0.65, 0.03])
        x_pan_slider_ax = plt.axes([0.1, 0.05, 0.65, 0.03])
        y_pan_slider_ax = plt.axes([0.1, 0.01, 0.65, 0.03])
        
        time_slider = Slider(
            ax=time_slider_ax,
            label='Time (hours)',
            valmin=min_time,
            valmax=max_time,
            valinit=initial_time
        )
        
        zoom_slider = Slider(
            ax=zoom_slider_ax,
            label='Zoom',
            valmin=0.1,
            valmax=4.0,
            valinit=1.0
        )
        
        x_pan_slider = Slider(
            ax=x_pan_slider_ax,
            label='X Pan',
            valmin=-5,
            valmax=5,
            valinit=0
        )
        
        y_pan_slider = Slider(
            ax=y_pan_slider_ax,
            label='Y Pan',
            valmin=-20,
            valmax=20,
            valinit=0
        )
        
        # Connect all sliders
        time_slider.on_changed(update)
        zoom_slider.on_changed(update)
        x_pan_slider.on_changed(update)
        y_pan_slider.on_changed(update)
        
        # Initial plot
        update(initial_time)
        
        plt.show()
        
    except Exception as e:
        print(f"Error in plotting: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    plot_trades() 