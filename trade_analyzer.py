from py_vollib.black_scholes_merton.implied_volatility import implied_volatility as iv_bsm
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Any
from market_data import MarketData
from curve_fitting import NewtonGaussInterpolator
from scipy.interpolate import interp1d
from joblib import Parallel, delayed

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
    midpoint: float = None  # New field for processed midpoint

class Operation:
    """Base class for all operations following the Command pattern."""
    def execute(self, trades: List[Trade], window_size: int, position: str):
        raise NotImplementedError("Execute method must be implemented by subclasses.")

class NewtonGaussOperation(Operation):
    """Performs Newton-Gauss interpolation on trade data."""
    def __init__(self):
        self.interpolator = None

    def execute(self, trades: List[Trade], window_size: int, position: str):
        # Extract relative strikes and IVs
        relative_strikes = np.array([(t.strike / t.spot_price - 1) * 100 for t in trades])
        ivs = np.array([t.iv for t in trades])
        
        # Fit Newton-Gauss interpolator
        self.interpolator = NewtonGaussInterpolator(relative_strikes, ivs)
        return self.interpolator

class MidpointTradeRuleOperation(Operation):
    """Applies midpoint trade rule on trade data."""
    def execute(self, trades: List[Trade], window_size: int, position: str):
        # Calculate midpoints
        strike_data = {}
        for trade in trades:
            rel_strike = (trade.strike / trade.spot_price - 1) * 100
            if rel_strike not in strike_data:
                strike_data[rel_strike] = []
            strike_data[rel_strike].append(trade.iv)
        
        midpoints = {strike: np.median(ivs) for strike, ivs in strike_data.items()}
        return midpoints

class TradeAnalyzer:
    def __init__(self, root: str, date: str):
        self.market_data = MarketData(root, date)
        self.trades: List[Trade] = []
        self.operations: List[Operation] = []
        self.processed_data: Dict[str, Any] = {}
    
    def load_data(self, trade_type: str = 'trade_quote'):
        """Load necessary trade data."""
        self.market_data.load_spot_prices()
        if trade_type == 'trade_quote':
            data = self.market_data.get_day_trade_quotes()
        elif trade_type == 'trade':
            data = self.market_data.get_day_trades()
        else:
            print("Invalid trade_type specified.")
            return
        
        if not data:
            print("No data loaded.")
            return
        
        # Get header indices
        header_format = data["header"]["format"]
        TIME_IDX = header_format.index("ms_of_day")
        PRICE_IDX = header_format.index("price")
        SIZE_IDX = header_format.index("size")
        BID_IDX = header_format.index("bid") if "bid" in header_format else None
        ASK_IDX = header_format.index("ask") if "ask" in header_format else None
        
        spot_times = sorted(self.market_data.spot_price_data.keys())
        
        # Process each response
        processed_trades = []
        for response in data["response"]:
            expiry = response["contract"]["expiration"]
            if str(expiry) != str(self.market_data.date):
                continue
                
            strike = response["contract"]["strike"] / 1000
            
            for tick in response["ticks"]:
                ms_of_day = tick[TIME_IDX]
                rounded_ms = round(ms_of_day / 500) * 500
                spot_price = self.market_data.get_spot_price(rounded_ms)
                
                if not spot_price:
                    continue
                
                is_buyer = False
                if BID_IDX is not None and ASK_IDX is not None:
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
                    bid=tick[BID_IDX] if BID_IDX is not None else None,
                    ask=tick[ASK_IDX] if ASK_IDX is not None else None,
                    is_buyer=is_buyer
                )
                trade.iv = self.calculate_iv(trade)
                if trade.iv is not None:
                    processed_trades.append(trade)
        
        self.trades = sorted(processed_trades, key=lambda x: x.time)
        print(f"Loaded {len(self.trades)} valid trades.")
    
    def calculate_iv(self, trade: Trade) -> float:
        """Calculate implied volatility for a trade."""
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
    
    def process_implied_volatilities(self):
        """Process all trade data to compute implied volatilities."""
        # Parallel processing can be utilized here if needed
        self.trades = Parallel(n_jobs=-1)(delayed(self.calculate_iv)(trade) for trade in self.trades)
        self.trades = [trade for trade in self.trades if trade.iv is not None]
        print(f"Processed implied volatilities for {len(self.trades)} trades.")
    
    def calculate_midpoints(self):
        """Process implied volatilities into midpoints."""
        strike_data = {}
        for trade in self.trades:
            rel_strike = (trade.strike / trade.spot_price - 1) * 100
            if rel_strike not in strike_data:
                strike_data[rel_strike] = []
            strike_data[rel_strike].append(trade.iv)
        
        for trade in self.trades:
            rel_strike = (trade.strike / trade.spot_price - 1) * 100
            trade.midpoint = np.median(strike_data[rel_strike])
        print("Calculated midpoints for all trades.")
    
    def add_operation(self, operation: Operation):
        """Add an operation to be applied to the trades."""
        self.operations.append(operation)
    
    def apply_operations(self, window_size: int, position: str = 'latest'):
        """
        Apply all added operations to the trades data.
        
        Parameters:
        - window_size: The size of the window to apply the operation.
        - position: 'latest' or 'middle' indicating the window's focus point.
        """
        for operation in self.operations:
            result = operation.execute(self.trades, window_size, position)
            self.processed_data[type(operation).__name__] = result
            print(f"Applied operation: {type(operation).__name__}")
    
    def plot_data(self, method: str = 'live_iv'):
        """
        Plot the processed data.
        
        Parameters:
        - method: 'live_iv' or 'cumulative_flow'
        """
        if method == 'live_iv':
            self.plot_live_iv()
        elif method == 'cumulative_flow':
            self.plot_cumulative_flow()
        else:
            print("Invalid plotting method specified.")
    
    def plot_live_iv(self):
        """Plot LIVE IV curve fit method."""
        plt.figure(figsize=(15, 8))
        for operation_name, data in self.processed_data.items():
            if isinstance(data, NewtonGaussInterpolator):
                x_smooth = np.linspace(min(data.x_data), max(data.x_data), 200)
                y_smooth = [data(x) for x in x_smooth]
                plt.plot(x_smooth, y_smooth * 100, label=f"{operation_name} Fit")
        
        plt.xlabel('% Away from Spot')
        plt.ylabel('Implied Volatility (%)')
        plt.title('LIVE IV Curve Fit')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_cumulative_flow(self):
        """Plot CUMULATIVE histogram flows method for classification."""
        plt.figure(figsize=(15, 8))
        net_flow = {}
        for trade in self.trades:
            rel_strike = (trade.strike / trade.spot_price - 1) * 100
            flow = trade.size if trade.is_buyer else -trade.size
            net_flow[rel_strike] = net_flow.get(rel_strike, 0) + flow
        
        relative_strikes = np.array(sorted(net_flow.keys()))
        flows = np.array([net_flow[k] for k in sorted(net_flow.keys())])
        
        width = 0.15
        colors = ['red' if f < 0 else 'green' for f in flows]
        plt.bar(relative_strikes, flows, width=width, color=colors, alpha=0.6)
        plt.axvline(x=0, color='blue', linestyle='--', alpha=0.5, label='Spot Price')
        plt.xlabel('% Away from Spot')
        plt.ylabel('Net Flow (Contracts)')
        plt.title('Cumulative Net Flow by Strike')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# Example Usage
if __name__ == "__main__":
    analyzer = TradeAnalyzer(root="SPXW", date="20250103")
    analyzer.load_data(trade_type='trade')  # or 'trade'
    analyzer.process_implied_volatilities()
    analyzer.calculate_midpoints()
    
    # Add operations
    # newton_op = NewtonGaussOperation()
    midpoint_op = MidpointTradeRuleOperation()
    
    # analyzer.add_operation(newton_op)
    analyzer.add_operation(midpoint_op)
    
    # Apply operations with a window size of 100 and focusing on the latest data
    analyzer.apply_operations(window_size=100, position='latest')
    
    # Plot results
    analyzer.plot_data(method='live_iv')
    analyzer.plot_data(method='cumulative_flow') 