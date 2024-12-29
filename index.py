import sys
from PyQt5 import QtWidgets
from gui import StreamWindow
from market_data import MarketDataService
from websocket_manager import WebSocketManager
from stream_manager import StreamManager
import json

def main():
    """Main application entry point"""
    app = QtWidgets.QApplication(sys.argv)
    
    # Create main window
    window = StreamWindow()
    window.resize(1600, 900)
    window.show()
    
    # Create WebSocket manager
    websocket_manager = WebSocketManager()
    
    # Initialize with some active contracts
    active_contracts = []
    market_data = MarketDataService()
    
    # Get initial contracts for SPY
    symbol = 'SPY'
    print(f"Getting contracts for {symbol}...")
    data = market_data.list_option_contracts(symbol)
    if data and 'response' in data:
        active_contracts = data['response']
        print(f"Found {len(active_contracts)} contracts")
    else:
        print("No contracts found, check if Theta Terminal is running")
        return app.exec_()
    
    # Create stream manager
    stream_manager = StreamManager(active_contracts, websocket_manager)
    
    # Connect signals
    stream_manager.trade_received.connect(window.handle_trade)
    stream_manager.message_received.connect(window.vol_surface_chart.update_quote)
    stream_manager.status_changed.connect(window.update_status)
    
    # Start WebSocket and streaming
    websocket_manager.start()
    stream_manager.start()
    
    # Clean up on exit
    app.aboutToQuit.connect(stream_manager.stop)
    
    return app.exec_()

if __name__ == '__main__':
    main() 