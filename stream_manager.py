from PyQt5.QtCore import QObject, pyqtSignal
from datetime import datetime, date
import json
from websocket_manager import WebSocketManager

class StreamManager(QObject):
    """Manages streaming of quotes and trades"""
    trade_received = pyqtSignal(dict, dict)  # contract, trade
    message_received = pyqtSignal(dict)  # message data
    status_changed = pyqtSignal(str)  # connection status
    
    def __init__(self, active_contracts, websocket_manager):
        super().__init__()
        self.active_contracts = active_contracts
        self.websocket_manager = websocket_manager
        
        # Connect websocket signals
        self.websocket_manager.message_received.connect(self._handle_message)
        self.websocket_manager.connected.connect(lambda: self.status_changed.emit("Connected"))
        self.websocket_manager.disconnected.connect(lambda: self.status_changed.emit("Disconnected"))
        self.websocket_manager.error_occurred.connect(lambda err: self.status_changed.emit(f"Error: {err}"))
        
    def _handle_message(self, message):
        """Handle incoming websocket message"""
        try:
            if not isinstance(message, dict):
                return
                
            message_type = message.get('type')
            if message_type == 'quote':
                self.message_received.emit(message)
            elif message_type == 'trade':
                contract = message.get('contract')
                trade = message.get('trade')
                if contract and trade:
                    self.trade_received.emit(contract, trade)
                    
        except Exception as e:
            print(f"Error handling message: {str(e)}")
            
    def start(self):
        """Start streaming for active contracts"""
        if not self.active_contracts:
            print("No active contracts to stream")
            return
            
        print(f"Starting stream for {len(self.active_contracts)} contracts")
        # Subscribe to quotes and trades for active contracts
        for contract in self.active_contracts:
            self.websocket_manager.subscribe_quotes(contract)
            self.websocket_manager.subscribe_trades(contract)
            
    def stop(self):
        """Stop streaming"""
        self.websocket_manager.stop()