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

def plot_trades():
    # Initialize and load data
    md = market_data.MarketData()
    target_date = "20241220"
    
    print("Loading spot price data...")
    md.load_spot_prices("SPY", target_date)
    if not md.spot_price_data:
        print("Failed to load spot price data")
        return
    
    print("Loading trade data...")
    trade_data = market_data.MarketData.get_day_trades("SPY", target_date)
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
    
    # Collect ALL trades with their timestamps
    all_trades = []
    
    print(f"Number of responses: {len(trade_data['response'])}")
    

    for response in trade_data["response"]:
        expiry = response["contract"]["expiration"]
        if str(expiry) != target_date:
            continue
            
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
                spot_price=spot_price
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
    
    # Create figure and subplots
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(bottom=0.3)  # Make more room for multiple sliders
    
    # Get time range and calculate initial axis limits
    min_time = all_trades[0][1].time
    max_time = all_trades[-1][1].time
    initial_time = min_time
    
    # Calculate initial axis limits from all trades
    all_strikes = [t.strike for _, t in all_trades]
    all_ivs = [t.iv * 100 for _, t in all_trades]
    all_spots = [t.spot_price for _, t in all_trades]
    
    x_min = min(min(all_strikes), min(all_spots)) * 0.99
    x_max = max(max(all_strikes), max(all_spots)) * 1.01
    y_min = min(all_ivs) * 0.5
    y_max = max(all_ivs) * 1.1
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    def update(val):
        current_time = time_slider.val
        zoom_level = zoom_slider.val
        x_pan = x_slider.val
        y_pan = y_slider.val
        
        # Find index of current time in all_trades
        current_idx = next(i for i, (ms, t) in enumerate(all_trades) if t.time >= current_time)
        
        # Get trades in window
        start_idx = max(0, current_idx - 1000)
        end_idx = min(len(all_trades), current_idx + 1000)
        window_trades = [t for _, t in all_trades[start_idx:end_idx]]
        
        # Clear previous plot
        ax.clear()
        
        # Plot all trades in black
        ax.scatter([t.strike for t in window_trades], 
                  [t.iv * 100 for t in window_trades],
                  s=20,
                  color='black',
                  alpha=0.5)
        
        # Get latest spot price for current time
        if window_trades:
            latest_spot = window_trades[-1].spot_price
            ax.axvline(x=latest_spot, color='blue', linestyle='--', alpha=0.5, 
                      label=f'Spot: {latest_spot:.0f}')
        
        # Fixed center points
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Calculate view ranges based on zoom
        view_x_range = x_range / zoom_level
        view_y_range = y_range / zoom_level
        
        # Calculate limits with zoom and pan
        x_left = center_x - view_x_range/2 + (x_pan * x_range/2)
        x_right = center_x + view_x_range/2 + (x_pan * x_range/2)
        y_bottom = center_y - view_y_range/2 + (y_pan * y_range/2)
        y_top = center_y + view_y_range/2 + (y_pan * y_range/2)
        
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(y_bottom, y_top)
        
        # Configure axes
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Implied Volatility (%)')
        window_start_time = window_trades[0].time if window_trades else current_time
        window_end_time = window_trades[-1].time if window_trades else current_time
        ax.set_title(f'Option Trades ({window_start_time:.2f} - {window_end_time:.2f} hours)\n'
                     f'Showing {len(window_trades)} trades')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        fig.canvas.draw_idle()
    
    # Create sliders
    time_slider_ax = plt.axes([0.1, 0.20, 0.65, 0.03])
    zoom_slider_ax = plt.axes([0.1, 0.15, 0.65, 0.03])
    x_slider_ax = plt.axes([0.1, 0.10, 0.65, 0.03])
    y_slider_ax = plt.axes([0.1, 0.05, 0.65, 0.03])
    
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
        valmax=10,
        valinit=1.0
    )
    
    x_slider = Slider(
        ax=x_slider_ax,
        label='X Pan',
        valmin=-1,
        valmax=1,
        valinit=0.0
    )
    
    y_slider = Slider(
        ax=y_slider_ax,
        label='Y Pan',
        valmin=-1,
        valmax=1,
        valinit=0.0
    )
    
    # Connect all sliders to update function
    time_slider.on_changed(update)
    zoom_slider.on_changed(update)
    x_slider.on_changed(update)
    y_slider.on_changed(update)
    
    # Initial plot
    update(initial_time)
    plt.show()

if __name__ == "__main__":
    plot_trades() 