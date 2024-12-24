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
    def classify_trade(trade_price: float, bid: float, ask: float, next_bid: float, next_ask: float, bid_size: int, ask_size: int, next_bid_size: int, next_ask_size: int) -> str:
        """
        Classify a trade as BUY or SELL based on price proximity to bid/ask and changes in quote sizes.
        Args:
            trade_price: float - The price at which the trade occurred
            bid: float - The bid price before the trade
            ask: float - The ask price before the trade
            next_bid: float - The bid price after the trade
            next_ask: float - The ask price after the trade
            bid_size: int - The bid size before the trade
            ask_size: int - The ask size before the trade
            next_bid_size: int - The bid size after the trade
            next_ask_size: int - The ask size after the trade
        Returns:
            str: 'BUY' if closer to ask, 'SELL' if closer to bid, 'UNKNOWN' if no quotes
        """
        if not bid or not ask:  # If we don't have quote data
            return "UNKNOWN"
        
        # Step 1: Check if the trade price matches the ask or bid
        if trade_price == ask and ask_size > next_ask_size:
            return "BUY"
        elif trade_price == bid and bid_size > next_bid_size:
            return "SELL"
        
        # Step 2: Check if the trade price is closer to the ask or bid
        mid_price = (bid + ask) / 2
        
        # Step 3: Use the next quote to determine if the trade was a buy or sell
        if next_ask < ask and next_bid == bid:
            return "BUY"
        elif next_bid > bid and next_ask == ask:
            return "SELL"
        
        # Step 4: Hidden order logic
        if trade_price > 0.7 * ask + 0.3 * bid:
            return "BUY"
        elif trade_price < 0.3 * ask + 0.7 * bid:
            return "SELL"
        
        # Step 5: Tick-test using midpoint comparison
        next_mid_price = (next_bid + next_ask) / 2
        if next_mid_price > mid_price:
            return "BUY"
        elif next_mid_price < mid_price:
            return "SELL"
        
        # Default to unknown if all else fails
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