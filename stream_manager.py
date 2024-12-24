import websockets.sync.client as websockets
import json
from typing import List, Dict
from PyQt5 import QtCore
from option_chain import OptionContract, OptionChainService
from datetime import datetime, date
import time

class WebSocketManager:
    def __init__(self):
        self.websocket = None
        self.running = True
        self.callbacks = []

    def connect(self):
        try:
            self.websocket = websockets.connect('ws://127.0.0.1:25520/v1/events')
            return True
        except Exception as e:
            return False

    def send(self, message):
        if self.websocket:
            self.websocket.send(json.dumps(message))

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def remove_callback(self, callback):
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def close(self):
        self.running = False
        if self.websocket:
            self.websocket.close()

class StreamManager(QtCore.QThread):
    message_received = QtCore.pyqtSignal(dict)
    status_changed = QtCore.pyqtSignal(str)
    error_occurred = QtCore.pyqtSignal(str)
    stream_count_changed = QtCore.pyqtSignal(int, int)
    trade_received = QtCore.pyqtSignal(dict, dict)  # contract, trade with classification

    def __init__(self, contracts: List[OptionContract], websocket_manager: WebSocketManager):
        super().__init__()
        self.all_contracts = contracts
        self.contracts = contracts
        self.running = True
        self.websocket_manager = websocket_manager
        self.active_streams = set()
        
        # Quote tracking for trade classification
        self.quote_history = {}  # {contract_key: [list of (timestamp, quote) tuples]}
        self.history_window = 5  # Keep last 5 quotes for each contract

    def get_quote_context(self, contract_key: str, trade_time: float) -> tuple:
        """Get quotes before and after trade"""
        quotes = self.quote_history.get(contract_key, [])
        print(f"\nQuote history for {contract_key}:")
        for qt, q in quotes:
            print(f"Time: {qt}, Bid: {q.get('bid')}, Ask: {q.get('ask')}")
        
        if not quotes:  # No quotes available
            return {}, {}
        
        # Sort quotes by time
        quotes.sort(key=lambda x: x[0])
        
        # If we only have one quote, use it for both prev and next
        if len(quotes) == 1:
            return quotes[0][1], quotes[0][1]
        
        # Find the most recent quote before the trade
        prev_quote = None
        next_quote = None
        
        # Use the most recent quote before the trade
        for qt, q in reversed(quotes):
            if qt <= trade_time:
                prev_quote = q
                break
        
        # If we didn't find a previous quote, use the earliest quote
        if not prev_quote:
            prev_quote = quotes[0][1]
        
        # Use the next available quote after the trade
        for qt, q in quotes:
            if qt > trade_time:
                next_quote = q
                break
        
        # If we didn't find a next quote, use the latest quote
        if not next_quote:
            next_quote = quotes[-1][1]
        
        print(f"Found prev_quote: {prev_quote}")
        print(f"Found next_quote: {next_quote}")
        return prev_quote, next_quote

    @staticmethod
    def classify_trade(price: float, size: int, prev_quote: dict, next_quote: dict) -> str:
        """Classify trade as BUY or SELL using quote context"""
        if not prev_quote and not next_quote:
            return 'UNKNOWN'
        
        # Use either quote if one is missing
        quote = prev_quote or next_quote
        
        # Extract quote data
        bid = quote.get('bid', 0)
        ask = quote.get('ask', float('inf'))
        
        # Simple classification based on price relative to quotes
        if price <= bid:
            return 'SELL'  # Hitting the bid
        elif price >= ask:
            return 'BUY'   # Lifting the offer
        
        # For prices between bid and ask, classify based on which side it's closer to
        mid_price = (bid + ask) / 2 if ask > bid else price
        return 'BUY' if price >= mid_price else 'SELL'

    def run(self):
        try:
            # Initial subscriptions
            for i, contract in enumerate(self.contracts):
                stream_key = f"{contract.root}_{contract.expiration}_{contract.strike}_{contract.right}"
                
                # Subscribe to both trades and quotes (need quotes for classification)
                trade_request = OptionChainService.create_stream_request(contract, i*2, "TRADE")
                quote_request = OptionChainService.create_stream_request(contract, i*2+1, "QUOTE")
                
                self.websocket_manager.send(trade_request)
                self.websocket_manager.send(quote_request)
                
                self.active_streams.add(stream_key)

            self.stream_count_changed.emit(len(self.active_streams), len(self.active_streams))

            while self.running:
                try:
                    response = self.websocket_manager.websocket.recv()
                    data = json.loads(response)
                    
                    if 'contract' not in data:
                        continue
                        
                    contract_key = OptionChainService.get_contract_key(data['contract'])
                    current_time = time.time()
                    
                    if 'quote' in data:
                        quote = data['quote']
                        print(f"\nReceived quote for {contract_key}:")
                        print(f"Raw quote data: {quote}")  # Let's see the full quote
                        
                        # Only store quotes for trade classification
                        if contract_key not in self.quote_history:
                            self.quote_history[contract_key] = []
                        
                        self.quote_history[contract_key].append((current_time, quote))
                        
                        if len(self.quote_history[contract_key]) > self.history_window:
                            self.quote_history[contract_key].pop(0)
                    
                    if 'trade' in data:
                        print(f"\nProcessing trade for {contract_key}:")
                        print(f"Trade data: {data['trade']}")  # Let's see the full trade
                        print(f"Available quotes: {len(self.quote_history.get(contract_key, []))}")
                        
                        # Get quote context
                        prev_quote, next_quote = self.get_quote_context(contract_key, current_time)
                        
                        # Classify trade
                        trade_data = data['trade'].copy()
                        trade_data['side'] = self.classify_trade(
                            float(trade_data['price']),
                            int(trade_data['size']),
                            prev_quote,
                            next_quote
                        )
                        
                        print(f"Classified as: {trade_data['side']}")
                        
                        # Emit classified trade
                        self.trade_received.emit(data['contract'], trade_data)

                except Exception as e:
                    self.error_occurred.emit(f"Message error: {str(e)}")
                    print(f"Error processing message: {str(e)}")
                    import traceback
                    traceback.print_exc()

        except Exception as e:
            self.error_occurred.emit(f"Connection error: {str(e)}")
            self.status_changed.emit("Error")