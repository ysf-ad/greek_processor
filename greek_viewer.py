import requests
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from matplotlib.widgets import Slider

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

def update(val):
    try:
        current_time = time_slider.val
        
        # Clear the plot
        ax.clear()
        
        # Find all data points up to current time
        valid_data = {}
        for strike in all_data:
            strike_data = [d for d in all_data[strike] if d.time <= current_time]
            if strike_data:
                valid_data[strike] = strike_data[-1]  # Take the latest data point
        
        if valid_data:
            # Get the latest spot price
            latest_spot = next(iter(valid_data.values())).spot_price
            
            # Plot IV curve
            strikes = sorted(valid_data.keys())
            rel_strikes = [(k/latest_spot - 1) * 100 for k in strikes]  # Convert to % away from spot
            ivs = [valid_data[k].implied_vol * 100 for k in strikes]  # Convert to percentage
            
            # Plot scatter points
            ax.scatter(rel_strikes, ivs, s=20, alpha=0.6)
            
            # Add vertical line at current spot price
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label=f'Spot: {latest_spot:.2f}')
            
            # Configure plot
            ax.set_xlabel('% Away from Spot')
            ax.set_ylabel('Implied Volatility (%)')
            ax.set_title(f'Implied Volatility Curve ({current_time:.2f} hours)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Set reasonable axis limits
            ax.set_xlim(-20, 20)
            y_min = max(0, min(ivs) - 10)  # Add some padding but keep above 0
            y_max = min(100, max(ivs) + 10)  # Add some padding but cap at 100
            ax.set_ylim(y_min, y_max)
        
        fig.canvas.draw_idle()
            
    except Exception as e:
        print(f"Error in update: {str(e)}")
        import traceback
        traceback.print_exc() 

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

def plot_vol_surface(root: str, target_date: int):
    global time_slider, ax, fig, all_data
    
    print(f"Fetching contracts for {root} on {target_date}...")
    contracts = get_0dte_contracts(root, target_date)
    
    if not contracts:
        print("No contracts found")
        return
    
    # Collect data for all strikes
    all_data = {}  # strike -> List[GreekData]
    
    # Separate calls and puts
    calls = [c for c in contracts if c["right"] == "C"]
    puts = [c for c in contracts if c["right"] == "P"]
    
    # Process calls first
    for contract in calls:
        strike = contract["strike"]
        right = contract["right"]
        
        print(f"Fetching data for {strike} {right}...")
        greek_data = fetch_greeks(
            root=root,
            expiry=target_date,
            strike=strike,
            right=right,
            start_date=target_date,
            end_date=target_date
        )
        
        if greek_data:
            all_data[strike/1000] = greek_data  # Convert strike to actual value
    
    if not all_data:
        print("No valid data collected")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(bottom=0.2)  # Make room for slider
    
    # Calculate time range
    all_times = [d.time for data in all_data.values() for d in data]
    min_time = min(all_times)
    max_time = max(all_times)
    
    # Create time slider
    time_slider_ax = plt.axes([0.1, 0.05, 0.65, 0.03])
    time_slider = Slider(
        ax=time_slider_ax,
        label='Time (hours)',
        valmin=min_time,
        valmax=max_time,
        valinit=min_time
    )
    
    # Connect slider
    time_slider.on_changed(update)
    
    # Initial plot
    update(min_time)
    
    plt.show()

if __name__ == "__main__":
    # Example usage
    target_date = 20250103  # Use integer format for date
    plot_vol_surface("SPXW", target_date) 