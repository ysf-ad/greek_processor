from PyQt5.QtCore import QObject, pyqtSignal
import websocket
import json
from threading import Thread

class WebSocketManager(QObject):
    """Manages WebSocket connection and message handling"""
    message_received = pyqtSignal(dict)
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.websocket = None
        self.running = True
        self.worker_thread = None
        
    def start(self):
        """Start WebSocket connection"""
        try:
            # Connect to Theta Terminal WebSocket
            self.websocket = websocket.WebSocketApp(
                "ws://127.0.0.1:25510/v2/stream",
                on_message=lambda ws, msg: self._on_message(msg),
                on_error=lambda ws, err: self._on_error(err),
                on_close=lambda ws, code, msg: self._on_close(),
                on_open=lambda ws: self._on_open()
            )
            
            # Start WebSocket in a separate thread
            self.worker_thread = Thread(target=self.websocket.run_forever)
            self.worker_thread.daemon = True
            self.worker_thread.start()
            
        except Exception as e:
            self.error_occurred.emit(f"Error starting WebSocket: {str(e)}")
            
    def subscribe_quotes(self, contract):
        """Subscribe to quotes for a contract"""
        try:
            if not self.websocket:
                return
                
            message = {
                "type": "subscribe",
                "message": {
                    "contract": contract,
                    "feed": "quote"
                }
            }
            self.websocket.send(json.dumps(message))
        except Exception as e:
            self.error_occurred.emit(f"Error subscribing to quotes: {str(e)}")
            
    def subscribe_trades(self, contract):
        """Subscribe to trades for a contract"""
        try:
            if not self.websocket:
                return
                
            message = {
                "type": "subscribe",
                "message": {
                    "contract": contract,
                    "feed": "trade"
                }
            }
            self.websocket.send(json.dumps(message))
        except Exception as e:
            self.error_occurred.emit(f"Error subscribing to trades: {str(e)}")
            
    def _on_message(self, message):
        """Internal message handler"""
        try:
            data = json.loads(message)
            self.message_received.emit(data)
        except Exception as e:
            self.error_occurred.emit(f"Error processing message: {str(e)}")
            
    def _on_error(self, error):
        """Internal error handler"""
        self.error_occurred.emit(f"WebSocket error: {str(error)}")
        
    def _on_close(self):
        """Internal close handler"""
        self.disconnected.emit()
        print("WebSocket connection closed")
        
    def _on_open(self):
        """Internal open handler"""
        self.connected.emit()
        print("WebSocket connection opened")
        
    def stop(self):
        """Stop WebSocket connection"""
        self.running = False
        if self.websocket:
            self.websocket.close()
        if self.worker_thread:
            self.worker_thread.join(timeout=1) 