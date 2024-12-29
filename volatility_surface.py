import numpy as np
from scipy.stats import norm
from scipy.optimize import bisect
from dataclasses import dataclass
from datetime import datetime, date
import requests
from market_data import MarketDataService
from PyQt5.QtCore import QTimer, QObject, pyqtSignal, QThread, pyqtSlot, QMutex
import time
from collections import defaultdict

@dataclass
class QuotePoint:
    strike: float
    expiry: date
    bid_vol: float
    ask_vol: float
    timestamp: datetime = None

class QuoteBufferWorker(QThread):
    """Worker thread for buffering quotes"""
    buffer_updated = pyqtSignal(dict)  # Emit the buffer when it's ready

    def __init__(self):
        super().__init__()
        self.quote_buffer = {}
        self.mutex = QMutex()
        self.running = True

    def add_quote(self, contract, quote_data):
        """Thread-safe quote addition to buffer"""
        try:
            root = contract['root']
            strike = float(contract['strike'])
            if strike > 1000:
                strike = strike / 1000.0
            expiry = datetime.strptime(str(contract['expiration']), '%Y%m%d').date()
            right = contract['right']
            
            # Create unique key for this contract
            key = (root, expiry, strike, right)
            
            # Thread-safe buffer update
            self.mutex.lock()
            self.quote_buffer[key] = (contract, quote_data)
            self.mutex.unlock()
            
        except Exception:
            pass

    def run(self):
        """Periodically emit the buffer"""
        while self.running:
            time.sleep(1.0)  # Wait for 1 second
            
            # Thread-safe buffer access
            self.mutex.lock()
            if self.quote_buffer:
                current_buffer = self.quote_buffer.copy()
                self.quote_buffer.clear()
                self.buffer_updated.emit(current_buffer)
            self.mutex.unlock()

    def stop(self):
        """Stop the worker thread"""
        self.running = False
        self.wait()


class BatchedVolatilitySurface(QObject):
    """Handles batched processing of quotes for IV surface calculation"""
    surface_updated = pyqtSignal(str, str)  # root, expiry

    def __init__(self, update_interval_ms=1000):
        super().__init__()
        self.processed_quotes = defaultdict(dict)  # {root: {(strike, expiry): QuotePoint}}
        self.market_data = MarketDataService()
        self.spot_prices = {}
        self.last_update = time.time()
        
        # Setup timer for periodic snapshots
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_snapshot)
        self.timer.start(update_interval_ms)

    def update_snapshot(self):
        """Get and process a new snapshot of quotes"""
        try:
            now = time.time()
            if now - self.last_update < 1.0:  # Only update once per second
                return

            # Get contracts for each root we're tracking
            roots = set(self.processed_quotes.keys())
            if not roots:  # If no roots yet, start with SPY
                roots = {'SPY'}

            for root in roots:
                # Get list of available contracts
                data = self.market_data.list_option_contracts(root)
                if not data or 'response' not in data:
                    print(f"No contracts found for {root}")
                    continue

                # Get spot price
                spot_price = self.market_data.get_current_spot_price(root)
                if not spot_price:
                    print(f"No spot price found for {root}")
                    continue
                self.spot_prices[root] = spot_price

                # Group contracts by expiry
                grouped_contracts = defaultdict(list)
                for contract in data['response']:
                    try:
                        strike = float(contract['strike'])
                        if strike > 1000:
                            strike = strike / 1000.0  # Convert back to decimal
                        expiry = datetime.strptime(str(contract['expiration']), '%Y%m%d').date()
                        right = contract['right']
                        
                        # Get quote for this contract
                        quote_data = self.market_data.get_quote(contract)
                        if not quote_data or 'response' not in quote_data:
                            continue
                            
                        quote = quote_data['response']
                        if not quote:  # Skip if no quote data
                            continue
                            
                        # Extract bid/ask from quote
                        bid = float(quote.get('bid', 0))
                        ask = float(quote.get('ask', 0))
                        
                        if bid <= 0 or ask <= 0 or ask < bid:
                            continue
                            
                        grouped_contracts[expiry].append((strike, right, {'bid': bid, 'ask': ask}))
                    except Exception as e:
                        print(f"Error processing contract: {str(e)}")
                        continue

                # Process each expiry
                for expiry, quotes in grouped_contracts.items():
                    if self.process_expiry_group(root, expiry, quotes, spot_price):
                        self.surface_updated.emit(root, expiry.strftime("%Y%m%d"))

                print(f"Processed {len(grouped_contracts)} expiries for {root}")

            self.last_update = now

        except Exception as e:
            print(f"Error updating snapshot: {str(e)}")

    def process_expiry_group(self, root, expiry, quotes, spot_price):
        """Process a group of quotes"""
        T = (expiry - date.today()).days / 365
        if T <= 0:
            return False

        # Group by strike
        strike_quotes = defaultdict(dict)
        for strike, right, quote in quotes:
            try:
                bid = float(quote.get('bid', 0))
                ask = float(quote.get('ask', 0))
                
                if bid <= 0 or ask <= 0 or ask < bid:
                    continue
                    
                spread = (ask - bid) / ((ask + bid) / 2)
                if spread > 0.15:  # Skip wide spreads
                    continue
                    
                strike_quotes[strike][right] = (bid, ask, spread)
            except:
                continue

        # Process each strike
        processed_any = False
        for strike, rights in strike_quotes.items():
            try:
                # Use put or call with tighter spread
                if 'C' in rights and 'P' in rights:
                    if rights['C'][2] <= rights['P'][2]:  # Compare spreads
                        bid, ask, _ = rights['C']
                        is_call = True
                    else:
                        bid, ask, _ = rights['P']
                        is_call = False
                elif 'C' in rights:
                    bid, ask, _ = rights['C']
                    is_call = True
                elif 'P' in rights:
                    bid, ask, _ = rights['P']
                    is_call = False
                else:
                    continue

                # Calculate IVs
                bid_vol = self.black_scholes_implied_vol(spot_price, strike, T, 0.0, bid, is_call=is_call)
                ask_vol = self.black_scholes_implied_vol(spot_price, strike, T, 0.0, ask, is_call=is_call)

                if (bid_vol and ask_vol and 
                    0.05 <= bid_vol <= 2.0 and 
                    0.05 <= ask_vol <= 2.0 and
                    ask_vol >= bid_vol):
                    self.processed_quotes[root][(strike, expiry)] = QuotePoint(
                        strike=strike,
                        expiry=expiry,
                        bid_vol=bid_vol,
                        ask_vol=ask_vol,
                        timestamp=datetime.now()
                    )
                    processed_any = True
            except:
                continue

        return processed_any

    def get_surface_data(self, root, selected_expiry):
        """Get processed surface data for visualization"""
        if root not in self.processed_quotes:
            return None

        # Convert selected expiry to date object
        target_expiry = datetime.strptime(selected_expiry, "%Y%m%d").date()
        spot_price = self.spot_prices.get(root)
        if not spot_price:
            return None

        # Get all quotes for selected expiry
        expiry_quotes = []
        for (strike, expiry), quote in self.processed_quotes[root].items():
            if expiry == target_expiry:
                expiry_quotes.append((strike, quote.bid_vol, quote.ask_vol))

        if not expiry_quotes:
            return None

        # Sort by strike
        expiry_quotes.sort(key=lambda x: x[0])
        strikes = [q[0] for q in expiry_quotes]
        bid_vols = [q[1] for q in expiry_quotes]
        ask_vols = [q[2] for q in expiry_quotes]
        moneyness = [np.log(k/spot_price) for k in strikes]

        return (moneyness, bid_vols, ask_vols, strikes),

    def get_expiries(self, root):
        """Get all available expiries for a root"""
        if root not in self.processed_quotes:
            return []
        expiries = sorted(set(expiry.strftime("%Y%m%d") 
                            for _, expiry in self.processed_quotes[root].keys()))
        return expiries

    def black_scholes_implied_vol(self, S, K, T, r, price, is_call=True):
        """Calculate implied volatility using Black-Scholes model"""
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