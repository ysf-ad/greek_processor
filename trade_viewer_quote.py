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

def fit_polynomial(x, y, trades, degree=4):
    """
    Fit a polynomial with outlier removal for the bottom 15% of strikes.
    """
    if len(x) < degree + 1:
        return np.polyfit(x, y, min(len(x)-1, 2))
    
    # Group data by strike
    strike_data = {}
    for i, (xi, yi) in enumerate(zip(x, y)):
        if xi not in strike_data:
            strike_data[xi] = []
        strike_data[xi].append(yi)
    
    # Find bottom 15% of strikes
    sorted_strikes = sorted(strike_data.keys())
    num_low_strikes = max(1, int(len(sorted_strikes) * 0.15))
    low_strikes = set(sorted_strikes[:num_low_strikes])
    
    # Process data for fitting
    filtered_x = []
    filtered_y = []
    
    for strike in sorted_strikes:
        ivs = strike_data[strike]
        if strike in low_strikes and len(ivs) > 3:
            # Remove outliers for bottom 15% of strikes
            ivs_array = np.array(ivs)
            q15, q85 = np.percentile(ivs_array, [15, 85])
            mask = (ivs_array >= q15) & (ivs_array <= q85)
            filtered_x.extend([strike] * np.sum(mask))
            filtered_y.extend(ivs_array[mask])
        else:
            # Keep all points for other strikes
            filtered_x.extend([strike] * len(ivs))
            filtered_y.extend(ivs)
    
    if len(filtered_x) < degree + 1:
        return np.polyfit(x, y, min(len(filtered_x)-1, degree))
    
    # Convert to numpy arrays and fit polynomial
    filtered_x = np.array(filtered_x)
    filtered_y = np.array(filtered_y)
    return np.polyfit(filtered_x, filtered_y, degree)

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
            
            # Calculate all relative strikes and IVs first
            relative_strikes = np.array([(t.strike/latest_spot - 1) * 100 for t in window_trades])
            ivs = np.array([t.iv for t in window_trades])
            
            # Calculate midpoint IVs
            mid_ivs = []
            unique_strikes = sorted(set(t.strike for t in window_trades))
            
            # Get latest quotes for each strike
            latest_quotes = {}
            for t in window_trades[::-1]:  # Reverse to get latest
                if t.strike not in latest_quotes:
                    latest_quotes[t.strike] = t
            
            # Calculate IVs for midpoint
            quote_strikes = []
            for strike in unique_strikes:
                if strike in latest_quotes:
                    quote = latest_quotes[strike]
                    # Create temporary trade for midpoint
                    mid_price = (quote.bid + quote.ask) / 2
                    mid_trade = Trade(
                        time=quote.time,
                        ms_of_day=quote.ms_of_day,
                        price=mid_price,
                        size=quote.size,
                        right=quote.right,
                        strike=quote.strike,
                        spot_price=quote.spot_price
                    )
                    
                    mid_iv = calculate_iv(mid_trade)
                    
                    if mid_iv is not None:
                        quote_strikes.append((strike/latest_spot - 1) * 100)
                        mid_ivs.append(mid_iv)
            
            # Separate buyer and seller trades for plotting
            buyer_mask = np.array([t.is_buyer for t in window_trades])
            seller_mask = ~buyer_mask
            
            # Plot trades with smaller dots
            ax.scatter(relative_strikes[buyer_mask], ivs[buyer_mask] * 100, 
                      s=1, color='green', alpha=0.3, label='Buyer Initiated')
            ax.scatter(relative_strikes[seller_mask], ivs[seller_mask] * 100, 
                      s=1, color='red', alpha=0.3, label='Seller Initiated')
            
            # Plot midpoint IV
            if quote_strikes:
                ax.plot(quote_strikes, np.array(mid_ivs) * 100, 'gray', linestyle='--', 
                       alpha=0.5, linewidth=0.5, label='Quote Midpoint')
            
            # Fit weighted polynomial
            coeffs = fit_polynomial(relative_strikes, ivs, window_trades)
            
            # Create smooth curve for polynomial
            x_smooth = np.linspace(min(relative_strikes), max(relative_strikes), 100)
            y_smooth = np.polyval(coeffs, x_smooth) * 100
            
            # Plot polynomial curve (thinner)
            ax.plot(x_smooth, y_smooth, 'b-', linewidth=0.75, alpha=0.5,
                   label='Polynomial Fit')
            
            # Vertical line at ATM
            ax.axvline(x=0, color='blue', linestyle='--', alpha=0.3, 
                      label=f'ATM (Spot: {latest_spot:.0f})')
            
            # Calculate view bounds
            x_center = 0  # Center on ATM
            y_center = np.mean(ivs) * 100  # Center on mean IV
            
            # Base ranges (before zoom)
            x_range = 15
            y_range = 20  # ±10 vol points
            
            # Apply zoom and pan
            x_min = x_center - (x_range / zoom) + x_pan
            x_max = x_center + (x_range / zoom) + x_pan
            y_min = y_center - (y_range / zoom) + y_pan
            y_max = y_center + (y_range / zoom) + y_pan
            
            # Configure plot
            ax.set_xlabel('% Away from Spot')
            ax.set_ylabel('Implied Volatility (%)')
            ax.set_title(f'Option Trades ({current_time:.2f} hours)')
            ax.grid(True, alpha=0.2)
            ax.legend(loc='upper right', fontsize=8)
            
            # Set axis limits
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(max(0, y_min), y_max)
            
            # Make sure axes are visible
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