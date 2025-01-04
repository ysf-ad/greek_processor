import market_data
import matplotlib.pyplot as plt
import numpy as np
import datetime
from datetime import UTC
from dataclasses import dataclass
from typing import List, Tuple

today = datetime.datetime.now().strftime('%Y%m%d')

def ms_to_hours(ms):
    """Convert milliseconds to hours"""
    return ms / (1000 * 3600)  # ms -> seconds -> hours

@dataclass
class Trade:
    time: float  # Hours since market open
    price: float
    size: int
    right: str  # 'P' or 'C'
    bid: float
    ask: float
    strike: float

class TradeClassifier:
    def __init__(self, trades: List[Trade]):
        self.trades = trades
        
    def classify_trades(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify trades as buyer or seller initiated based on proximity to bid/ask
        Returns: Tuple of (call_premium, put_premium) arrays
        """
        times = np.array([t.time for t in self.trades])
        sort_idx = np.argsort(times)
        
        # Pre-allocate arrays
        call_premium = np.zeros(len(self.trades))
        put_premium = np.zeros(len(self.trades))
        
        for i in sort_idx:
            trade = self.trades[i]
            # Determine if buyer or seller initiated
            mid_price = (trade.bid + trade.ask) / 2
            is_buy = trade.price >= mid_price
            
            # Calculate signed premium (positive for buys, negative for sells)
            if is_buy:
                premium = trade.price * trade.size  # Buyer pays premium
            else:
                premium = -trade.price * trade.size  # Seller receives premium
            
            if trade.right == 'C':
                call_premium[i] = premium
            else:  # Put
                put_premium[i] = premium
        
        return call_premium, put_premium

    @staticmethod
    def get_aggressor(trade: Trade) -> str:
        """
        Determine if trade was buyer or seller initiated
        Returns: 'BUY' or 'SELL'
        """
        mid_price = (trade.bid + trade.ask) / 2
        return 'BUY' if trade.price >= mid_price else 'SELL'
    
    @staticmethod
    def get_premium(trade: Trade) -> float:
        """Calculate signed premium based on aggressor"""
        is_buy = TradeClassifier.get_aggressor(trade) == 'BUY'
        premium = trade.price * trade.size
        return premium if is_buy else -premium

    def calculate_net_premiums(trades: List[Trade]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate net premiums for puts and calls"""
        times = np.array([t.time for t in trades])
        sort_idx = np.argsort(times)
        
        call_premium = np.zeros(len(trades))
        put_premium = np.zeros(len(trades))
        
        for i in sort_idx:
            trade = trades[i]
            premium = TradeClassifier.get_premium(trade)
            
            if trade.right == 'C':
                call_premium[i] = premium
            else:  # Put
                put_premium[i] = premium
        
        return call_premium, put_premium

    @staticmethod
    def get_size_flow(trade: Trade) -> float:
        """Calculate signed size based on aggressor"""
        is_buy = TradeClassifier.get_aggressor(trade) == 'BUY'
        return trade.size if is_buy else -trade.size

    @staticmethod
    def calculate_net_flow(trades: List[Trade]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate net size flow for puts and calls"""
        times = np.array([t.time for t in trades])
        sort_idx = np.argsort(times)
        
        call_flow = np.zeros(len(trades))
        put_flow = np.zeros(len(trades))
        
        for i in sort_idx:
            trade = trades[i]
            size_flow = TradeClassifier.get_size_flow(trade)
            
            if trade.right == 'C':
                call_flow[i] = size_flow
            else:  # Put
                put_flow[i] = size_flow
        
        return call_flow, put_flow

if __name__ == "__main__":
    trade_data = market_data.MarketData.get_day_trade_quotes("SPXW", "20241230")
    if trade_data is None:
        print("Error: No data received")
        exit()
    print("received data")

    # Create Trade objects
    trades = []
    for response in trade_data["response"]:
        strike = response["contract"]["strike"] / 1000
        for tick in response["ticks"]:
            trades.append(Trade(
                time=ms_to_hours(tick[0]),  # Use new time conversion
                price=tick[1],
                size=tick[7],
                right=response["contract"]["right"],
                bid=tick[2],
                ask=tick[3],
                strike=strike
            ))

    # Create figure with one subplot and twin y-axis
    fig, ax1 = plt.subplots(figsize=(15, 8))
    ax2 = ax1.twinx()  # Create second y-axis for premium

    # Get top 300 trades by size
    big_trades = sorted(trades, key=lambda x: x.size, reverse=True)[:300]
    
    # Plot top 300 trades scatter
    for trade in big_trades:
        color = 'green' if trade.right == 'C' else 'red'
        aggressor = TradeClassifier.get_aggressor(trade)
        marker = '^' if aggressor == 'BUY' else 'v'
        ax1.scatter(trade.time, trade.strike, s=trade.size/2, 
                   color=color, alpha=0.5, marker=marker)

    # Calculate net premium (using ALL trades)
    call_premium, put_premium = TradeClassifier.calculate_net_flow(trades)
    
    # Calculate cumulative premium
    times_array = np.array([t.time for t in trades])
    sort_idx = np.argsort(times_array)
    times_array = times_array[sort_idx]
    cum_call_premium = np.cumsum(call_premium[sort_idx])
    cum_put_premium = np.cumsum(put_premium[sort_idx])
    
    # Plot premium on second y-axis
    ax2.plot(times_array, cum_call_premium, 'g-', label='Call Net Premium', alpha=0.7, linewidth=2)
    ax2.plot(times_array, cum_put_premium, 'r-', label='Put Net Premium', alpha=0.7, linewidth=2)
    
    # Configure axes
    ax1.set_ylabel('Strike Price')
    ax1.set_title('SPX Option Trades')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Hours')
    
    ax2.set_ylabel('Net Premium ($)')
    ax2.legend(loc='upper right')

    # Set x-axis limits to show market hours
    ax1.set_xlim(0, 6.5)  # 9:30 to 16:00

    plt.tight_layout()
    plt.show()

