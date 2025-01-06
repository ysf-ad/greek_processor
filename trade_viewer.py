import market_data
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility as iv_bsm
from matplotlib.widgets import Slider

@dataclass
class Trade:
    time: float
    ms_of_day: int
    price: float
    size: int
    right: str
    strike: float
    spot_price: float
    expiry: int
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

def update(val):
    try:
        current_time = time_slider.val
        
        # Clear the plot
        ax.clear()
        
        # Get unique expiries
        expiries = sorted(list(set(t.expiry for t in all_trades)))
        
        # Create a color map for different expiries
        colors = plt.cm.rainbow(np.linspace(0, 1, len(expiries)))
        
        for expiry, color in zip(expiries, colors):
            # Find trades for this expiry
            expiry_trades = [t for t in all_trades if t.expiry == expiry]
            
            try:
                # Find current index in these trades
                current_idx = next(i for i, t in enumerate(expiry_trades) if t.time >= current_time)
                start_idx = max(0, current_idx - 100)
                end_idx = min(len(expiry_trades), current_idx + 100)
                window_trades = expiry_trades[start_idx:end_idx]
                
                if window_trades:
                    # Get the latest spot price
                    latest_spot = window_trades[-1].spot_price
                    
                    # Calculate relative strikes and IVs
                    relative_strikes = [(t.strike/latest_spot - 1) * 100 for t in window_trades]
                    ivs = [t.iv * 100 for t in window_trades]
                    
                    # Plot trades
                    ax.scatter(relative_strikes, ivs, s=20, alpha=0.6,
                             c=['green' if t.right == 'C' else 'red' for t in window_trades],
                             marker='o' if expiry == expiries[0] else '^',  # Different marker for each expiry
                             label=f'Trades (Exp: {expiry})')
                    
                    # Fit polynomial
                    coeffs = np.polyfit(relative_strikes, ivs, 3)
                    x_smooth = np.linspace(min(relative_strikes), max(relative_strikes), 100)
                    y_smooth = np.polyval(coeffs, x_smooth)
                    ax.plot(x_smooth, y_smooth, '-', color=color, linewidth=2, alpha=0.7,
                           label=f'Fit (Exp: {expiry})')
            except StopIteration:
                # No trades after current_time for this expiry
                continue
        
        # Add vertical line at current spot price (using the last valid spot price)
        if 'latest_spot' in locals():
            ax.axvline(x=0, color='blue', linestyle='--', alpha=0.5, label=f'Spot: {latest_spot:.2f}')
        
        # Configure plot
        ax.set_xlabel('% Away from Spot')
        ax.set_ylabel('Implied Volatility (%)')
        ax.set_title(f'Option Trades ({current_time:.2f} hours)')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set reasonable axis limits
        ax.set_xlim(-20, 20)
        ax.set_ylim(0, 100)
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        fig.canvas.draw_idle()
            
    except Exception as e:
        print(f"Error in update: {str(e)}")
        import traceback
        traceback.print_exc()

def plot_trades():
    global time_slider, ax, fig, all_trades
    
    # Initialize and load data
    md = market_data.MarketData()
    target_date = "20250103"
    root = "SPXW"
    
    print("Loading spot price data...")
    md.load_spot_prices(root, target_date)
    if not md.spot_price_data:
        print("Failed to load spot price data")
        return
    
    print("Loading trade data...")
    trade_data = market_data.MarketData.get_day_trades(root, target_date)
    if not trade_data:
        print("No trades found")
        return
    
    # Get indices once
    header_format = trade_data["header"]["format"]
    TIME_IDX = header_format.index("ms_of_day")
    PRICE_IDX = header_format.index("price")
    SIZE_IDX = header_format.index("size")
    
    # Pre-process spot price data
    spot_times = sorted(md.spot_price_data.keys())
    
    # Collect all trades
    all_trades = []
    
    print(f"Number of responses: {len(trade_data['response'])}")
    
    for response in trade_data["response"]:
        expiry = response["contract"]["expiration"]
        strike = response["contract"]["strike"] / 1000
        
        for tick in response["ticks"]:
            ms_of_day = tick[TIME_IDX]
            rounded_ms = round(ms_of_day / 500) * 500
            spot_price = md.spot_price_data.get(rounded_ms, md.spot_price_data[spot_times[0]])
            
            trade = Trade(
                time=ms_of_day / (1000 * 3600),
                ms_of_day=ms_of_day,
                price=tick[PRICE_IDX],
                size=tick[SIZE_IDX],
                right=response["contract"]["right"],
                strike=strike,
                spot_price=spot_price,
                expiry=expiry
            )
            trade.iv = calculate_iv(trade)
            
            if trade.iv is not None:
                all_trades.append(trade)
    
    # Sort trades by time
    all_trades.sort(key=lambda x: x.time)
    print(f"Collected {len(all_trades)} valid trades")
    
    if not all_trades:
        print("No valid trades to plot")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(bottom=0.2)  # Make room for slider
    
    # Calculate time range
    min_time = min(t.time for t in all_trades)
    max_time = max(t.time for t in all_trades)
    initial_time = min_time
    
    # Create time slider
    time_slider_ax = plt.axes([0.1, 0.05, 0.65, 0.03])
    time_slider = Slider(
        ax=time_slider_ax,
        label='Time (hours)',
        valmin=min_time,
        valmax=max_time,
        valinit=initial_time
    )
    
    # Connect slider
    time_slider.on_changed(update)
    
    # Initial plot
    update(initial_time)
    
    plt.show()

if __name__ == "__main__":
    plot_trades() 
    