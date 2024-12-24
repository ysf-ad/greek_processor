import numpy as np
from scipy.stats import norm
from scipy.optimize import bisect
from dataclasses import dataclass
from datetime import datetime, date
import requests
from market_data import MarketDataService

@dataclass
class QuotePoint:
    strike: float
    expiry: date
    implied_vol: float
    timestamp: datetime = None

class VolatilitySurface:
    def __init__(self, target_date=None):
        self.quotes = {}  # {root: {(strike, expiry): QuotePoint}}
        self.market_data = MarketDataService()
        self.spot_prices = {}  # Cache spot prices
        self.last_spot_update = {}  # Track when we last updated spot prices
        self.target_date = target_date or date.today().strftime("%Y%m%d")
        self.today_expiry = date.today().strftime("%Y%m%d")  # Today's expiry for testing
        print(f"Initialized VolatilitySurface with date: {self.target_date}")
        
    def estimate_spot_from_quotes(self, root):
        """Estimate spot price from ATM options"""
        if root not in self.quotes:
            return None
            
        # Get all quotes for the target expiry
        quotes = self.quotes[root]
        expiries = sorted(set(k[1] for k in quotes.keys()))
        if not expiries:
            return None
            
        target_expiry = expiries[0]
        
        # Get all strikes for this expiry
        strikes = sorted(set(k[0] for k, v in quotes.items() if k[1] == target_expiry))
        if not strikes:
            return None
            
        # Find the strike with the smallest bid-ask spread
        best_spread = float('inf')
        best_strike = None
        
        for strike in strikes:
            call_key = (strike, target_expiry, 'C')
            put_key = (strike, target_expiry, 'P')
            
            if call_key in quotes and put_key in quotes:
                call_vol = quotes[call_key].implied_vol
                put_vol = quotes[put_key].implied_vol
                spread = abs(call_vol - put_vol)
                
                if spread < best_spread:
                    best_spread = spread
                    best_strike = strike
        
        return best_strike if best_strike is not None else None
        
    def get_spot_price(self, root):
        """Get spot price with caching"""
        now = datetime.now()
        
        # Update spot price every second
        if (root not in self.last_spot_update or 
            (now - self.last_spot_update[root]).total_seconds() > 1.0):
            self.spot_prices[root] = self.market_data.get_current_spot_price(root)
            self.last_spot_update[root] = now
            print(f"Updated spot price for {root}: {self.spot_prices[root]}")
        
        return self.spot_prices.get(root, 590)  # Default to 590 if not available
        
    def update_quote(self, contract, quote_data):
        """Update quote data with raw implied vol"""
        root = contract['root']
        strike = float(contract['strike'])/1000
        expiry = datetime.strptime(str(contract['expiration']), '%Y%m%d').date()
        right = contract['right']
        
        # Initialize quotes dictionary for this root if needed
        if root not in self.quotes:
            self.quotes[root] = {}
        
        try:
            # Get bid/ask prices
            bid = float(quote_data.get('bid', 0))
            ask = float(quote_data.get('ask', 0))
            
            # Skip if invalid quotes
            if bid <= 0 or ask <= 0:
                print(f"Skipping due to invalid bid/ask: {bid}/{ask}")
                return
            
            # Skip if spread is too wide
            spread = (ask - bid) / ((ask + bid) / 2)
            if spread > 0.15:  # 15% spread filter
                print(f"Skipping due to wide spread: {spread:.2%}")
                return
            
            # Calculate mid price
            mid_price = (bid + ask) / 2
            
            # Get spot price
            spot_price = self.get_spot_price(root)
            if not spot_price:
                print("No spot price available")
                return
            
            # Calculate time to expiry
            T = (expiry - date.today()).days / 365
            if T <= 0:
                print("Expired option")
                return
            
            # Calculate implied vol
            implied_vol = self.black_scholes_implied_vol(spot_price, strike, T, 0.0, mid_price, is_call=(right=='C'))
            
            # Skip if implied vol is outside reasonable range
            if implied_vol is None or implied_vol < 0.05 or implied_vol > 2.0:
                print(f"Skipping due to invalid IV: {implied_vol}")
                return
            
            # Store the quote
            key = (strike, expiry, right)
            self.quotes[root][key] = QuotePoint(
                strike=strike,
                expiry=expiry,
                implied_vol=implied_vol,
                timestamp=datetime.now()
            )
            
            print(f"Stored quote for {root} {strike} {right}: IV={implied_vol:.2%}")
            
        except Exception as e:
            print(f"Error processing quote: {str(e)}")
            
    def get_expiries(self, root):
        """Get all unique expiries for a given root"""
        if root not in self.quotes:
            return []
        expiries = sorted(set(expiry.strftime("%Y%m%d") for _, expiry, _ in self.quotes[root].keys()))
        print(f"Available expiries for {root}: {expiries}")  # Debug print
        return expiries

    def get_surface_data(self, root, selected_expiry):
        """Get surface data for selected expiry"""
        if root not in self.quotes or not self.quotes[root]:
            print(f"No quotes for {root}")
            return None
        
        # Get quotes for the root
        spot_price = self.get_spot_price(root)
        print(f"Spot price for {root}: {spot_price}")
        
        # Convert selected expiry to date object
        target_expiry = datetime.strptime(selected_expiry, "%Y%m%d").date()
        print(f"Using expiry: {target_expiry}")
        
        # Group quotes by strike for selected expiry
        strike_quotes = {}  # {strike: {'C': vol, 'P': vol}}
        for (strike, expiry, right), quote in self.quotes[root].items():
            if expiry == target_expiry:  # Only process quotes for selected expiry
                if strike not in strike_quotes:
                    strike_quotes[strike] = {}
                strike_quotes[strike][right] = quote.implied_vol
        
        # Combine put and call IVs for each strike
        combined_points = []  # [(strike, vol)]
        for strike, vols in strike_quotes.items():
            if 'C' in vols and 'P' in vols:
                combined_vol = (vols['C'] + vols['P']) / 2
                combined_points.append((strike, combined_vol))
            elif 'C' in vols:
                combined_points.append((strike, vols['C']))
            elif 'P' in vols:
                combined_points.append((strike, vols['P']))
        
        if not combined_points:
            return None
            
        # Sort by strike
        combined_points.sort(key=lambda x: x[0])
        
        # Split into arrays
        strikes = [p[0] for p in combined_points]
        vols = [p[1] for p in combined_points]
        moneyness = [np.log(k/spot_price) for k in strikes]
        
        print(f"Combined {len(combined_points)} points for {root}")
        
        return (moneyness, vols, strikes),
        
    def black_scholes_implied_vol(self, S, K, T, r, price, is_call=True):
        def bs_price(vol):
            d1 = (np.log(S/K) + (r + 0.5*vol*vol)*T)/(vol*np.sqrt(T))
            d2 = d1 - vol*np.sqrt(T)
            if is_call:
                return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            else:
                return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
        def vol_objective(vol):
            return bs_price(vol) - price
            
        try:
            implied_vol = bisect(vol_objective, 0.0001, 5.0)
            return implied_vol
        except:
            return None

    def update_surface(self):
        """Update the volatility surface plot"""
        try:
            self.surface_ax.clear()
            
            # Get current root from selector
            root = self.root_selector.currentText()
            if not root:
                return
            
            # Debug prints
            print(f"\nUpdating surface plot for {root}...")
            
            surface_data, _ = self.vol_surface.get_surface_data(root)
            print(f"Surface data: {surface_data}")
            
            if surface_data and surface_data[0]:  # Check if we have moneyness data
                moneyness, vols, strikes = surface_data
                print(f"Plotting {len(moneyness)} points for {root}")
                print(f"Moneyness range: {min(moneyness):.3f} to {max(moneyness):.3f}")
                print(f"Vol range: {min(vols):.3f} to {max(vols):.3f}")
                
                # Plot vertical line at spot price (moneyness = 0)
                self.surface_ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Spot')
                
                # Plot scatter with different colors for puts and calls
                put_mask = [m < 0 for m in moneyness]  # Puts are typically when moneyness < 0
                call_mask = [m >= 0 for m in moneyness]
                
                # Plot puts in red, calls in blue
                self.surface_ax.scatter(
                    [m for i, m in enumerate(moneyness) if put_mask[i]], 
                    [v for i, v in enumerate(vols) if put_mask[i]], 
                    color='red', alpha=0.6, label='Puts'
                )
                self.surface_ax.scatter(
                    [m for i, m in enumerate(moneyness) if call_mask[i]], 
                    [v for i, v in enumerate(vols) if call_mask[i]], 
                    color='blue', alpha=0.6, label='Calls'
                )
                
                # Add strike labels for reference
                for i, (m, v, k) in enumerate(zip(moneyness, vols, strikes)):
                    if i % 3 == 0:  # Label every third point
                        self.surface_ax.annotate(f'{k:.1f}',
                                               (m, v),
                                               xytext=(0, 5),
                                               textcoords='offset points',
                                               ha='center',
                                               fontsize=8)
            else:
                print(f"No valid surface data for {root}")
                
            self.surface_ax.set_xlabel('Log Moneyness')
            self.surface_ax.set_ylabel('Implied Volatility')
            self.surface_ax.set_title(f'Volatility Surface - {root}')
            self.surface_ax.grid(True)
            self.surface_ax.legend()
            
            # Set reasonable y-axis limits
            self.surface_ax.set_ylim(0, 1.0)  # Typical IV range
            
            # Refresh the plot
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating surface: {str(e)}")
            self.log_error(f"Surface update error: {str(e)}")