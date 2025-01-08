import market_data
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from trade_viewer import Trade, calculate_iv
import seaborn as sns
from matplotlib.widgets import Slider
from scipy.stats import norm

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

def classify_trades_by_polynomial(trades: List[Trade], current_time: float, window_size: int = 100) -> Dict[float, int]:
    """
    Classify trades as buys/sells based on their position relative to the polynomial fit
    Returns a dictionary of strike -> net flow (positive for buys, negative for sells)
    """
    # Filter trades up to current time
    trades = [t for t in trades if t.time <= current_time]
    
    if not trades:
        return {}
    
    # Sort trades by time
    trades.sort(key=lambda x: x.time)
    
    # Initialize net flow dictionary
    net_flow = {}
    
    # Group trades by timestamp
    time_groups = {}
    for trade in trades:
        # Round to nearest millisecond to group trades at same timestamp
        rounded_time = round(trade.time * 1000) / 1000
        if rounded_time not in time_groups:
            time_groups[rounded_time] = []
        time_groups[rounded_time].append(trade)
    
    # Process each timestamp
    timestamps = sorted(time_groups.keys())
    for i, current_time in enumerate(timestamps):
        current_trades = time_groups[current_time]
        
        # Get window of trades centered on current timestamp
        window_start = max(0, i - window_size//2)
        window_end = min(len(timestamps), i + window_size//2)
        window_trades = []
        for t in timestamps[window_start:window_end]:
            window_trades.extend(time_groups[t])
        
        if len(window_trades) < 3:  # Need at least 3 points for polynomial fit
            continue
        
        # Get latest spot price
        latest_spot = window_trades[-1].spot_price
        
        # Calculate relative strikes and IVs for window
        relative_strikes = np.array([(t.strike/latest_spot - 1) * 100 for t in window_trades])
        ivs = np.array([t.iv * 100 for t in window_trades])
        
        # Fit polynomial once for this timestamp
        coeffs = fit_polynomial(relative_strikes, ivs, window_trades)
        
        # Classify all trades at this timestamp using the same polynomial
        for trade in current_trades:
            current_relative_strike = (trade.strike/latest_spot - 1) * 100
            poly_iv = np.polyval(coeffs, current_relative_strike)
            actual_iv = trade.iv * 100
            is_sell = actual_iv > poly_iv
            
            # Update net flow
            if trade.strike not in net_flow:
                net_flow[trade.strike] = 0
            net_flow[trade.strike] += (-1 if is_sell else 1) * trade.size
    
    return net_flow

class NetFlowPlotter:
    def __init__(self, trades: List[Trade]):
        self.trades = trades
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        plt.subplots_adjust(bottom=0.25)  # Make room for sliders
        
        # Get time range
        self.min_time = min(t.time for t in trades)
        self.max_time = max(t.time for t in trades)
        
        # Create sliders
        time_ax = plt.axes([0.1, 0.1, 0.65, 0.03])
        self.time_slider = Slider(
            time_ax, 'Time (hours)', 
            self.min_time, self.max_time,
            valinit=self.max_time
        )
        
        x_scale_ax = plt.axes([0.1, 0.05, 0.65, 0.03])
        self.x_scale_slider = Slider(
            x_scale_ax, 'X Scale', 
            1, 20, 
            valinit=5
        )
        
        # Connect sliders to update function
        self.time_slider.on_changed(self.update)
        self.x_scale_slider.on_changed(self.update)
        
        # Initial plot
        self.update(None)
        
    def update(self, _):
        current_time = self.time_slider.val
        x_scale = self.x_scale_slider.val
        
        # Clear current plot
        self.ax.clear()
        
        # Get net flow at current time
        net_flow = classify_trades_by_polynomial(self.trades, current_time)
        
        if not net_flow:
            return
        
        # Get latest spot price
        current_trades = [t for t in self.trades if t.time <= current_time]
        if not current_trades:
            return
        spot_price = current_trades[-1].spot_price
        
        # Create bar plot
        strikes = sorted(net_flow.keys())
        flows = [net_flow[k] for k in strikes]
        relative_strikes = [(k/spot_price - 1) * 100 for k in strikes]
        
        # Make bars thinner by adjusting width
        width = 0.15  # Thinner bars
        colors = ['red' if f < 0 else 'green' for f in flows]
        self.ax.bar(relative_strikes, flows, width=width, color=colors, alpha=0.6)
        
        # Add vertical line at spot price
        self.ax.axvline(x=0, color='blue', linestyle='--', alpha=0.5, label='Spot Price')
        
        # Configure plot
        self.ax.set_xlabel('% Away from Spot')
        self.ax.set_ylabel('Net Flow (Contracts)')
        self.ax.set_title(f'Net Flow by Strike at {current_time:.2f} hours')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # Set x-axis limits based on slider
        self.ax.set_xlim(-x_scale, x_scale)
        
        # Redraw
        self.fig.canvas.draw_idle()

def main():
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
        if str(expiry) != str(target_date):  # Only process 0DTE
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
    
    # Create interactive plot
    plotter = NetFlowPlotter(all_trades)
    plt.show()

if __name__ == "__main__":
    main() 