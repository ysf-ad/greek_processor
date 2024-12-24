from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import List, Dict
import heapq

@dataclass
class ContractConfig:
    symbols: List[str]
    min_oi: int
    max_contracts: int = None

@dataclass
class OptionContract:
    root: str
    expiration: str
    strike: str
    right: str
    bid_size: int
    ask_size: int

class OptionChainService:
    """Handles option chain filtering and processing"""
    
    @staticmethod
    def filter_active_contracts(data: Dict, config: ContractConfig) -> List[OptionContract]:
        """Filter and sort contracts based on quote data"""
        contracts = []
        
        for contract_data in data.get('response', []):
            try:
                contract = contract_data['contract']
                quote = contract_data['quote']
                
                # Skip if missing required fields
                if not all([contract.get('expiration'), contract.get('strike'), contract.get('right')]):
                    continue
                
                # Basic validity checks
                if quote['bid'] <= 0 or quote['ask'] <= 0:
                    continue
                    
                # Check if not expired
                try:
                    exp_date = datetime.strptime(str(contract['expiration']), '%Y%m%d')
                    if exp_date.date() > date.today():
                        contracts.append(OptionContract(
                            root=contract['root'],
                            expiration=str(contract['expiration']),
                            strike=str(contract['strike']),
                            right=contract['right'],
                            bid_size=quote['bid_size'],
                            ask_size=quote['ask_size']
                        ))
                except ValueError as e:
                    print(f"Skipping contract due to invalid date: {contract.get('expiration')}")
                    continue
                    
            except Exception as e:
                print(f"Error processing contract: {str(e)}")
                continue
        
        print(f"Filtered {len(contracts)} valid contracts")
        return contracts

    @staticmethod
    def create_stream_request(contract: OptionContract, stream_id: int, req_type: str = "TRADE") -> Dict:
        """Create websocket stream request for a contract"""
        return {
            'msg_type': 'STREAM',
            'sec_type': 'OPTION',
            'req_type': req_type,
            'add': True,
            'id': stream_id,
            'contract': {
                'root': contract.root,
                'expiration': contract.expiration,
                'strike': contract.strike,
                'right': contract.right
            }
        } 

    @staticmethod
    def classify_trade(trade_price: float, bid: float, ask: float, next_bid: float, next_ask: float, 
                      bid_size: int, ask_size: int, next_bid_size: int, next_ask_size: int) -> str:
        """Enhanced hidden order detection"""
        # Step 1: Exact quote matches with volume change
        if trade_price == ask and ask_size > next_ask_size:
            return "BUY"
        elif trade_price == bid and bid_size > next_bid_size:
            return "SELL"

        # Step 2: Quote movement detection
        if next_ask < ask and next_bid == bid:  # Ask moves down after trade
            return "BUY"
        elif next_bid > bid and next_ask == ask:  # Bid moves up after trade
            return "SELL"

        # Step 3: Enhanced hidden order logic
        spread = ask - bid
        relative_price = (trade_price - bid) / spread if spread > 0 else 0.5

        # Inside spread trades
        if bid < trade_price < ask:
            # Very close to quotes (within 10%)
            if relative_price > 0.9:  # Near ask
                return "BUY"
            elif relative_price < 0.1:  # Near bid
                return "SELL"
            
            # Volume pressure
            if next_ask_size < ask_size and next_bid_size == bid_size:
                return "BUY"  # Ask size decreased but bid didn't
            elif next_bid_size < bid_size and next_ask_size == ask_size:
                return "SELL"  # Bid size decreased but ask didn't
            
            # Price pressure
            if next_ask < ask:  # Ask moves down after trade
                return "BUY"
            elif next_bid > bid:  # Bid moves up after trade
                return "SELL"
            
            # Standard spread division (70/30)
            if relative_price > 0.7:
                return "BUY"
            elif relative_price < 0.3:
                return "SELL"

        # Step 4: Outside quotes trades
        if trade_price > ask:  # Above ask
            return "BUY"
        elif trade_price < bid:  # Below bid
            return "SELL"

        # Step 5: Tick test using midpoint comparison
        mid_price = (bid + ask) / 2
        next_mid_price = (next_bid + next_ask) / 2
        if next_mid_price > mid_price:
            return "BUY"
        elif next_mid_price < mid_price:
            return "SELL"

        return "UNKNOWN"

    @staticmethod
    def get_contract_key(contract: dict) -> str:
        """Create a unique key for a contract"""
        return f"{contract['root']}_{contract['expiration']}_{contract['strike']}_{contract['right']}"

    @staticmethod
    def format_quote(quote: dict) -> str:
        """Format quote data for display"""
        return f"B: {quote['bid']:.2f}x{quote['bid_size']} ({quote['bid_exchange']}) " \
               f"A: {quote['ask']:.2f}x{quote['ask_size']} ({quote['ask_exchange']})"