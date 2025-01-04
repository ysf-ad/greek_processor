from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

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
    @staticmethod
    def get_aggressor(trade: Trade) -> str:
        """
        Determine if trade was buyer or seller initiated
        Returns: 'BUY' or 'SELL'
        """
        mid_price = (trade.bid + trade.ask) / 2
        return 'BUY' if trade.price >= mid_price else 'SELL'
    
    @staticmethod
    def get_size_flow(trade: Trade) -> float:
        """Calculate signed size based on aggressor"""
        is_buy = TradeClassifier.get_aggressor(trade) == 'BUY'
        return trade.size if is_buy else -trade.size

    @staticmethod
    def calculate_net_flow(trades: List[Trade]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate net premium flow for puts and calls"""
        times = np.array([t.time for t in trades])
        sort_idx = np.argsort(times)
        
        call_premium = np.zeros(len(trades))
        put_premium = np.zeros(len(trades))
        
        for i in sort_idx:
            trade = trades[i]
            # Determine if buyer or seller initiated
            mid_price = (trade.bid + trade.ask) / 2
            is_buy = trade.price >= mid_price
            
            # Calculate premium in dollars (price * size * 100 for contract multiplier)
            premium = trade.price * trade.size * 100
            if not is_buy:
                premium = -premium  # Negative for sells
            
            if trade.right == 'C':
                call_premium[i] = premium
            else:  # Put
                put_premium[i] = premium
        
        return call_premium, put_premium
