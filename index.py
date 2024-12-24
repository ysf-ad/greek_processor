import sys
from PyQt5 import QtWidgets
from gui import StreamWindow
from market_data import MarketDataService
from option_chain import ContractConfig, OptionChainService
from stream_manager import StreamManager, WebSocketManager
import json

def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # Optional: Set target date for historical mode
    target_date = None  # Default to live/today
    
    # Configure what to stream
    config = ContractConfig(
        symbols=["SPY"],
        min_oi=100,
        max_contracts=14500 
    )
    
    # Create websocket manager first
    websocket_manager = WebSocketManager()
    if not websocket_manager.connect():
        print("Failed to connect to websocket")
        sys.exit(1)
    
    # Get market data and contracts
    market_data = MarketDataService()
    active_contracts = []
    for symbol in config.symbols:
        data = market_data.get_bulk_quote_data(symbol)
        contracts = OptionChainService.filter_active_contracts(data, config)
        active_contracts.extend(contracts)
    
    # Sort all contracts by quote volume and apply max limit
    active_contracts.sort(key=lambda x: (x.bid_size + x.ask_size), reverse=True)
    if config.max_contracts:
        active_contracts = active_contracts[:config.max_contracts]
    
    # Create stream manager
    stream_manager = StreamManager(active_contracts, websocket_manager)
    
    # Create window with stream manager reference
    window = StreamWindow(target_date, stream_manager)
    window.show()
    
    window.update_status("Connected")
    
    # Connect signals
    stream_manager.status_changed.connect(window.update_status)
    stream_manager.error_occurred.connect(window.log_error)
    stream_manager.stream_count_changed.connect(window.update_stream_count)
    stream_manager.trade_received.connect(lambda contract, trade: (
        window.trade_chart.add_trade(contract, trade['side']),
        window.handle_trade(contract, trade, {})
    ))
    
    # Start streaming
    stream_manager.start()
    
    # Make sure to close websocket on exit
    app.aboutToQuit.connect(websocket_manager.close)
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 