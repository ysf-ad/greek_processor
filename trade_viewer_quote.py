import market_data
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility as iv_bsm
from matplotlib.widgets import Slider
from scipy.optimize import minimize
import requests

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

@dataclass
class GreekData:
    time: float  # Hours
    ms_of_day: int
    bid: float
    ask: float
    delta: float
    theta: float
    vega: float
    rho: float
    epsilon: float
    lambda_: float
    implied_vol: float
    iv_error: float
    spot_price: float
    strike: float
    right: str

def fetch_greeks(root: str, expiry: int, strike: int, right: str, start_date: int, end_date: int, ivl: int = 90000) -> List[GreekData]:
    url = f"http://127.0.0.1:25510/v2/hist/option/greeks"
    params = {
        "root": root,
        "exp": expiry,
        "strike": strike,
        "right": right,
        "start_date": start_date,
        "end_date": end_date,
        "ivl": ivl
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if "response" not in data:
            print(f"No data for {root} {strike} {right}")
            return []
        
        # Get indices from format
        format_list = data["header"]["format"]
        ms_idx = format_list.index("ms_of_day")
        bid_idx = format_list.index("bid")
        ask_idx = format_list.index("ask")
        delta_idx = format_list.index("delta")
        theta_idx = format_list.index("theta")
        vega_idx = format_list.index("vega")
        rho_idx = format_list.index("rho")
        epsilon_idx = format_list.index("epsilon")
        lambda_idx = format_list.index("lambda")
        iv_idx = format_list.index("implied_vol")
        iv_error_idx = format_list.index("iv_error")
        spot_idx = format_list.index("underlying_price")
        
        greek_data = []
        for tick in data["response"]:
            if tick[iv_idx] > 0:  # Only include valid IV data
                greek_data.append(GreekData(
                    time=tick[ms_idx] / (1000 * 3600),  # Convert to hours
                    ms_of_day=tick[ms_idx],
                    bid=tick[bid_idx],
                    ask=tick[ask_idx],
                    delta=tick[delta_idx],
                    theta=tick[theta_idx],
                    vega=tick[vega_idx],
                    rho=tick[rho_idx],
                    epsilon=tick[epsilon_idx],
                    lambda_=tick[lambda_idx],
                    implied_vol=tick[iv_idx],
                    iv_error=tick[iv_error_idx],
                    spot_price=tick[spot_idx],
                    strike=strike/1000,  # Convert to actual strike
                    right=right
                ))
        
        return greek_data
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return []

def get_0dte_contracts(root: str, target_date: int) -> List[Dict]:
    """Get all 0DTE contracts for the given date"""
    url = f"http://127.0.0.1:25510/v2/list/contracts/option/quote"
    params = {
        "root": root,
        "start_date": target_date
    }
    
    try:
        print(f"Fetching contracts from: {url} with params: {params}")
        response = requests.get(url, params=params)
        data = response.json()
        
        if "response" not in data:
            print(f"No contracts found for {root} on {target_date}")
            return []
        
        print(f"Got {len(data['response'])} total contracts")
        
        # Get indices from format
        format_list = data["header"]["format"]
        root_idx = format_list.index("root")
        exp_idx = format_list.index("expiration")
        strike_idx = format_list.index("strike")
        right_idx = format_list.index("right")
        
        # Filter for contracts expiring on target_date
        contracts = []
        for contract in data["response"]:
            if contract[exp_idx] == target_date:  # Compare with target_date
                contracts.append({
                    "root": contract[root_idx],
                    "expiration": contract[exp_idx],
                    "strike": contract[strike_idx],
                    "right": contract[right_idx]
                })
        
        print(f"Found {len(contracts)} contracts expiring on {target_date}")
        if len(contracts) > 0:
            print(f"Sample contract: {contracts[0]}")
        return contracts
    except Exception as e:
        print(f"Error fetching contracts: {str(e)}")
        print(f"URL: {url}")
        print(f"Params: {params}")
        return []

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
        zoom = zoom_slider.val
        x_pan = x_pan_slider.val
        y_pan = y_pan_slider.val
        
        # Clear the plot
        ax.clear()
        
        # First plot greek data
        valid_greek_data = {}
        for strike in greek_data:
            strike_data = [d for d in greek_data[strike] if d.time <= current_time]
            if strike_data:
                valid_greek_data[strike] = strike_data[-1]  # Take the latest data point
        
        if valid_greek_data:
            latest_spot = next(iter(valid_greek_data.values())).spot_price
            
            # Plot greek IV curve
            strikes = sorted(valid_greek_data.keys())
            rel_strikes = [(k/latest_spot - 1) * 100 for k in strikes]
            ivs = [valid_greek_data[k].implied_vol * 100 for k in strikes]
            
            # Plot greek scatter points
            ax.scatter(rel_strikes, ivs, s=20, alpha=0.3, color='blue', label='Greeks')
        
        # Then plot trade data
        current_idx = next(i for i, (ms, t) in enumerate(all_trades) if t.time >= current_time)
        start_idx = max(0, current_idx - 100)
        end_idx = min(len(all_trades), current_idx + 100)
        window_trades = [t for _, t in all_trades[start_idx:end_idx]]
        
        if window_trades:
            latest_spot = window_trades[-1].spot_price
            
            # Calculate all relative strikes and IVs first
            relative_strikes = np.array([(t.strike/latest_spot - 1) * 100 for t in window_trades])
            ivs = np.array([t.iv for t in window_trades])
            
            # Separate buyer and seller trades for plotting
            buyer_mask = np.array([t.is_buyer for t in window_trades])
            seller_mask = ~buyer_mask
            
            # Plot trades
            ax.scatter(relative_strikes[buyer_mask], ivs[buyer_mask] * 100, 
                      s=2, color='green', alpha=0.5, label='Buyer Initiated')
            ax.scatter(relative_strikes[seller_mask], ivs[seller_mask] * 100, 
                      s=2, color='red', alpha=0.5, label='Seller Initiated')
            
            # Fit polynomial
            coeffs = fit_polynomial(relative_strikes, ivs)
            
            # Create smooth curve for polynomial
            x_smooth = np.linspace(min(relative_strikes), max(relative_strikes), 100)
            y_smooth = np.polyval(coeffs, x_smooth) * 100
            
            # Plot polynomial curve
            ax.plot(x_smooth, y_smooth, 'b-', linewidth=2, alpha=0.7,
                   label='Polynomial Fit')
            
            # Vertical line at ATM
            ax.axvline(x=0, color='blue', linestyle='--', alpha=0.5, 
                      label=f'ATM (Spot: {latest_spot:.0f})')
            
            # Calculate view bounds
            x_center = 0  # Center on ATM
            y_center = np.mean(ivs) * 100  # Center on mean IV
            
            # Base ranges (before zoom)
            x_range = 15  # Increased range to ±7.5% from spot
            y_range = 20  # ±10 vol points
            
            # Apply zoom and pan
            x_min = x_center - (x_range / zoom) + x_pan
            x_max = x_center + (x_range / zoom) + x_pan
            y_min = y_center - (y_range / zoom) + y_pan
            y_max = y_center + (y_range / zoom) + y_pan
            
            # Configure plot
            ax.set_xlabel('% Away from Spot')
            ax.set_ylabel('Implied Volatility (%)')
            ax.set_title(f'Option Trades and Greeks ({current_time:.2f} hours)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
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
    global time_slider, zoom_slider, x_pan_slider, y_pan_slider, ax, all_trades, greek_data, fig
    
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
    
    # Get greek data
    print("Fetching greek data...")
    contracts = get_0dte_contracts(root, target_date)
    greek_data = {}  # strike -> List[GreekData]
    
    # Process calls first
    for contract in contracts:
        if contract["right"] == "C":  # Only process calls for now
            strike = contract["strike"]
            right = contract["right"]
            
            print(f"Fetching greek data for {strike} {right}...")
            data = fetch_greeks(
                root=root,
                expiry=target_date,
                strike=strike,
                right=right,
                start_date=target_date,
                end_date=target_date
            )
            
            if data:
                greek_data[strike/1000] = data
    
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
            valmax=2.0,
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

def fit_polynomial(x, y, degree=3):
    """Fit a polynomial of given degree to the data."""
    return np.polyfit(x, y, degree)

if __name__ == "__main__":
    plot_trades() 