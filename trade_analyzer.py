from py_vollib.black_scholes_merton.implied_volatility import implied_volatility as iv_bsm
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
from market_data import MarketData
from curve_fitting import MLInterpolator
from matplotlib.widgets import Slider
import datetime

@dataclass
class Trade:
    time: float  # Hours since market open
    ms_of_day: int
    price: float
    size: int
    right: str  # 'P' or 'C'
    strike: float
    spot_price: float
    bid: float = None
    ask: float = None
    is_buyer: bool = None
    iv: float = None
    rel_strike: float = None

class TradeAnalyzer:
    def __init__(self, root: str, date: str):
        self.market_data = MarketData(root, date)
        self.trades: List[Trade] = []
        
    def calculate_iv(self, trade: Trade) -> float:
        """Calculate implied volatility for a trade."""
        try:
            # For 0DTE, use a minimum of 1 minute to expiry
            T = max((16 - trade.time) / 24 / 365, 1/24/365)
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
            
            return iv if 0 < iv < 2 else None  # Tighter bounds for 0DTE
        except:
            return None
    
    def load_data(self, trade_type: str = 'trade'):
        """Load and process trade data with optimization for 0DTE."""
        print("Loading spot prices...")
        self.market_data.load_spot_prices()
        
        print("Loading trade data...")
        data = self.market_data.get_day_trades() if trade_type == 'trade' else self.market_data.get_day_trade_quotes()
        
        if not data:
            print("No data loaded.")
            return
        
        header_format = data["header"]["format"]
        TIME_IDX = header_format.index("ms_of_day")
        PRICE_IDX = header_format.index("price")
        SIZE_IDX = header_format.index("size")
        
        processed_trades = []
        
        print(f"Processing responses for 0DTE trades...")
        for response in data["response"]:
            expiry = response["contract"]["expiration"]
            if str(expiry) != str(self.market_data.date):
                continue
                
            strike = response["contract"]["strike"] / 1000
            
            for tick in response["ticks"]:
                ms_of_day = tick[TIME_IDX]
                rounded_ms = round(ms_of_day / 500) * 500
                time = rounded_ms / (1000 * 3600)
                
                # Skip trades outside market hours (9:30 AM to 4:00 PM)
                if time < 9.5 or time > 16:
                    continue
                
                spot_price = self.market_data.get_spot_price(rounded_ms)
                if not spot_price:
                    continue
                
                trade = Trade(
                    time=time,
                    ms_of_day=rounded_ms,  # Use rounded ms
                    price=tick[PRICE_IDX],
                    size=tick[SIZE_IDX],
                    right=response["contract"]["right"],
                    strike=strike,
                    spot_price=spot_price,
                    is_buyer=True
                )
                
                trade.iv = self.calculate_iv(trade)
                if trade.iv is not None:
                    trade.rel_strike = (trade.strike / trade.spot_price - 1) * 100
                    processed_trades.append(trade)
        
        self.trades = sorted(processed_trades, key=lambda x: x.time)
        print(f"Processed {len(self.trades)} valid 0DTE trades")

    def plot_iv_surface(self):
        """Plot IV surface with improved visualization using trade-count based window."""
        if not self.trades:
            print("No trades to plot")
            return
            
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        plt.subplots_adjust(bottom=0.25)

        times = np.array([t.time for t in self.trades])
        rel_strikes = np.array([t.rel_strike for t in self.trades])
        ivs = np.array([t.iv for t in self.trades])

        # Create time-based scatter plot
        scatter = ax.scatter(rel_strikes, ivs, c='blue', alpha=0.5, s=2)
        line, = ax.plot([], [], 'r-', lw=2, label='Fitted Curve')

        ax.set_xlabel('Relative Strike (%)')
        ax.set_ylabel('Implied Volatility')
        ax.set_title('0DTE IV Surface')
        ax.grid(True)
        ax.legend()

        # Time slider (market hours: 9:30 AM to 4:00 PM)
        ax_time = plt.axes([0.1, 0.1, 0.65, 0.03])
        time_slider = Slider(ax_time, 'Time (hours)', 9.5, 16.0,
                           valinit=9.5, valstep=0.1)

        # Window size in number of trades
        ax_window = plt.axes([0.1, 0.05, 0.65, 0.03])
        window_slider = Slider(ax_window, 'Window (trades)', 50, 500,
                             valinit=500, valstep=50)

        def update(val):
            current_time = time_slider.val
            window_size = int(window_slider.val)

            # Find the index closest to current time
            current_idx = np.argmin(np.abs(times - current_time))
            
            # Get window of trades centered on current time
            start_idx = max(0, current_idx - window_size // 2)
            end_idx = min(len(times), start_idx + window_size)
            
            # Adjust start_idx if end_idx hit the limit
            if end_idx == len(times):
                start_idx = max(0, end_idx - window_size)

            current_strikes = rel_strikes[start_idx:end_idx]
            current_ivs = ivs[start_idx:end_idx]

            if len(current_strikes) > 5:  # Minimum points for fitting
                scatter.set_offsets(np.c_[current_strikes, current_ivs])

                try:
                    interpolator = MLInterpolator(current_strikes, current_ivs, smoothing_factor=0.3)
                    x_smooth = np.linspace(current_strikes.min(), current_strikes.max(), 100)
                    y_smooth = interpolator(x_smooth)
                    line.set_data(x_smooth, y_smooth)
                except Exception as e:
                    print(f"Fitting error: {e}")
                    line.set_data([], [])
            else:
                scatter.set_offsets(np.c_[[], []])
                line.set_data([], [])

            if len(current_strikes) > 0:
                ax.set_xlim(current_strikes.min() - 1, current_strikes.max() + 1)
                ax.set_ylim(max(0, current_ivs.min() - 0.1), min(2, current_ivs.max() + 0.1))
            
            # Update title with current time and number of trades
            ax.set_title(f'0DTE IV Surface - Time: {current_time:.1f} (Trades: {len(current_strikes)})')
            fig.canvas.draw_idle()

        time_slider.on_changed(update)
        window_slider.on_changed(update)
        update(None)

        plt.show()

if __name__ == "__main__":
    analyzer = TradeAnalyzer(root="SPXW", date="20250103")
    analyzer.load_data(trade_type='trade')  # 30-minute window
    analyzer.plot_iv_surface() 