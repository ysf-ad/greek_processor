import market_data
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from trade_viewer import Trade, calculate_iv
import seaborn as sns
from matplotlib.widgets import Slider
from scipy.stats import norm

def normalize_data(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Normalize data to improve polynomial fitting"""
    x_mean, x_std = np.mean(x), np.std(x)
    y_mean, y_std = np.mean(y), np.std(y)
    
    x_norm = (x - x_mean) / x_std if x_std > 0 else x - x_mean
    y_norm = (y - y_mean) / y_std if y_std > 0 else y - y_mean
    
    return x_norm, y_norm, x_mean, x_std, y_mean, y_std

def robust_polyfit(x: np.ndarray, y: np.ndarray, degree: int = 2) -> Tuple[np.ndarray, callable]:
    """Perform robust polynomial fitting with normalization"""
    try:
        # Normalize data
        x_norm, y_norm, x_mean, x_std, y_mean, y_std = normalize_data(x, y)
        
        # Fit polynomial to normalized data
        coeffs = np.polyfit(x_norm, y_norm, degree)
        
        # Create function to evaluate polynomial with denormalization
        def poly_func(x_new):
            x_new_norm = (x_new - x_mean) / x_std if x_std > 0 else x_new - x_mean
            y_new_norm = np.polyval(coeffs, x_new_norm)
            return y_new_norm * y_std + y_mean if y_std > 0 else y_new_norm + y_mean
        
        return coeffs, poly_func
    except Exception as e:
        print(f"Error in polynomial fitting: {e}")
        return None, lambda x: np.mean(y)

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
    
    # Process trades in sliding windows
    for i in range(len(trades)):
        # Get window of trades centered on current trade
        start_idx = max(0, i - window_size//2)
        end_idx = min(len(trades), i + window_size//2)
        window_trades = trades[start_idx:end_idx]
        
        if len(window_trades) < 3:  # Need at least 3 points for polynomial fit
            continue
        
        # Get current trade
        current_trade = trades[i]
        latest_spot = current_trade.spot_price
        
        # Calculate relative strikes and IVs for window
        relative_strikes = np.array([(t.strike/latest_spot - 1) * 100 for t in window_trades])
        ivs = np.array([t.iv * 100 for t in window_trades])
        
        # Fit polynomial
        _, poly_func = robust_polyfit(relative_strikes, ivs)
        
        # Get polynomial value at current trade's strike
        current_relative_strike = (current_trade.strike/latest_spot - 1) * 100
        poly_iv = poly_func(current_relative_strike)
        
        # Classify trade
        actual_iv = current_trade.iv * 100
        is_sell = actual_iv > poly_iv
        
        # Update net flow
        if current_trade.strike not in net_flow:
            net_flow[current_trade.strike] = 0
        net_flow[current_trade.strike] += (-1 if is_sell else 1) * current_trade.size
    
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