import sys
from datetime import datetime, date
from PyQt5 import QtWidgets, QtCore, QtChart
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QComboBox, QCheckBox
from PyQt5.QtChart import QChart, QChartView, QBarSet, QBarSeries, QBarCategoryAxis, QValueAxis
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor, QCursor
from PyQt5.QtWidgets import QToolTip
import json
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from volatility_surface import VolatilitySurface
from option_chain import OptionChainService

class VolatilitySurfaceChart(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        # Create horizontal layout for controls
        controls_layout = QHBoxLayout()
        
        # Add root selector
        self.root_selector = QComboBox()
        self.root_selector.currentTextChanged.connect(self.on_root_changed)
        controls_layout.addWidget(self.root_selector)
        
        # Add expiry selector
        self.expiry_selector = QComboBox()
        self.expiry_selector.currentTextChanged.connect(self.on_expiry_changed)
        controls_layout.addWidget(self.expiry_selector)
        
        # Add toggle for expiry filtering
        self.expiry_filter = QCheckBox("Stream All Expiries")
        self.expiry_filter.setToolTip("Enable streaming for all expiries (default: 0DTE only)")
        self.expiry_filter.stateChanged.connect(self.on_expiry_filter_changed)
        controls_layout.addWidget(self.expiry_filter)
        
        # Add controls to main layout
        self.layout.addLayout(controls_layout)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        self.surface_ax = self.figure.add_subplot(111)
        self.layout.addWidget(self.canvas)
        
        # Initialize volatility surface
        self.vol_surface = VolatilitySurface()
        print("VolatilitySurfaceChart initialized")
        
        # Debug label
        self.debug_label = QLabel("No quotes processed yet")
        self.layout.addWidget(self.debug_label)
        
        # Start update timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_surface)
        self.timer.start(1000)  # Update display every second
        
        # Get today's date in YYYYMMDD format
        self.today_expiry = date.today().strftime("%Y%m%d")
        
    def on_root_changed(self, root):
        """Handle root selection change"""
        if root:
            # Update expiry selector
            expiries = self.vol_surface.get_expiries(root)
            self.expiry_selector.clear()
            self.expiry_selector.addItems(expiries)
            if expiries:
                self.expiry_selector.setCurrentIndex(0)
            self.update_surface()
        
    def on_expiry_changed(self, expiry):
        """Handle expiry selection change"""
        if expiry and hasattr(self, 'stream_manager'):
            self.stream_manager.set_selected_expiry(expiry)
        self.update_surface()
        
    def update_surface(self):
        """Update the volatility surface plot"""
        try:
            self.surface_ax.clear()
            
            # Get current root and expiry from selectors
            root = self.root_selector.currentText()
            expiry = self.expiry_selector.currentText()
            if not root or not expiry:
                return
            
            # Debug prints
            print(f"\nUpdating surface plot for {root} with expiry {expiry}...")
            
            # Get surface data for selected expiry
            surface_data = self.vol_surface.get_surface_data(root, expiry)
            if surface_data and surface_data[0]:  # Check if we have moneyness data
                moneyness, vols, strikes = surface_data[0]  # Unpack the first tuple
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
            self.surface_ax.set_title(f'Volatility Surface - {root} ({expiry})')
            self.surface_ax.grid(True)
            self.surface_ax.legend()
            
            # Set reasonable y-axis limits
            self.surface_ax.set_ylim(0, 1.0)  # Typical IV range
            
            # Refresh the plot
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating surface: {str(e)}")
            self.log_error(f"Surface update error: {str(e)}")
            
    def update_surface_timer(self):
        """Timer callback to update the surface periodically"""
        self.update_surface()
        
    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)
        
    def log_error(self, error):
        """Log an error message"""
        print(f"ERROR: {error}")
        
    def update_surface_timer(self):
        """Timer callback to update the surface periodically"""
        self.update_surface()
        
    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)
        
    def log_error(self, error):
        """Log an error message"""
        print(f"ERROR: {error}")
        
    def update_quote(self, contract, quote):
        """Process quote and update the volatility surface"""
        try:
            root = contract['root']
            expiry = str(contract['expiration'])
            
            # Update root selector if needed
            if root not in [self.root_selector.itemText(i) for i in range(self.root_selector.count())]:
                self.root_selector.addItem(root)
                if not self.root_selector.currentText():  # If nothing selected, select this root
                    self.root_selector.setCurrentText(root)
            
            # Process the quote
            self.vol_surface.update_quote(contract, quote)
            
            # Always update expiry selector for the current root
            if root == self.root_selector.currentText():
                expiries = self.vol_surface.get_expiries(root)
                current_expiry = self.expiry_selector.currentText()
                
                # Update expiry dropdown while preserving selection
                self.expiry_selector.clear()
                self.expiry_selector.addItems(expiries)
                
                # Restore previous selection if it exists, otherwise select first item
                if current_expiry in expiries:
                    self.expiry_selector.setCurrentText(current_expiry)
                elif expiries:
                    self.expiry_selector.setCurrentIndex(0)
            
            # Update debug info
            strike = int(contract['strike']) / 1000
            right = contract['right']
            bid = quote.get('bid', 'N/A')
            ask = quote.get('ask', 'N/A')
            
            # Get current quote count for this root
            quote_count = 0
            if root in self.vol_surface.quotes:
                quote_count = len(self.vol_surface.quotes[root])
                print(f"Processed quote: {root} {strike} {right} {expiry} | "
                      f"B: {bid} A: {ask} | "
                      f"Total quotes: {quote_count}")
            
            # Only update debug label if this is for the selected root
            if root == self.root_selector.currentText():
                self.debug_label.setText(
                    f"Last quote: {root} {strike} {right} {expiry} | "
                    f"B: {bid} A: {ask} | "
                    f"Total quotes: {quote_count}"
                )
                
        except Exception as e:
            print(f"Error processing quote: {str(e)}")

    def on_expiry_filter_changed(self, state):
        """Handle expiry filter toggle"""
        root = self.root_selector.currentText()
        if not state:  # If unchecked, filter to only 0DTE
            if root in self.vol_surface.quotes:
                filtered_quotes = {}
                for key, quote in self.vol_surface.quotes[root].items():
                    strike, expiry, right = key
                    if expiry == datetime.strptime(self.today_expiry, "%Y%m%d").date():
                        filtered_quotes[key] = quote
                
                # Update quotes dictionary with filtered quotes
                self.vol_surface.quotes[root] = filtered_quotes
                print(f"Filtered to 0DTE only. Kept {len(filtered_quotes)} quotes.")
        
        # Update the surface
        self.update_surface()

class TradeActivityChart(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        # Contract selector
        self.contract_selector = QComboBox()
        self.contract_selector.currentTextChanged.connect(self.update_chart)
        self.layout.addWidget(self.contract_selector)
        
        # Create chart
        self.chart = QChart()
        self.chart.setAnimationOptions(QChart.NoAnimation)  # Disable animations
        self.chart.setTitle("Buy/Sell Activity by Strike")
        
        # Create chart view
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        
        # Enable tooltips
        self.chart_view.setToolTipDuration(-1)  # Keep tooltip visible while hovering
        
        self.layout.addWidget(self.chart_view)
        
        # Data storage
        self.trade_data = {}  # {root: {strike: {'buys': count, 'sells': count, 'prices': []}}}

    def add_trade(self, contract, side):
        root = contract['root']
        strike = int(contract['strike'])/1000
        price = float(contract.get('price', 0))
        
        if root not in self.trade_data:
            self.trade_data[root] = {}
            if root not in [self.contract_selector.itemText(i) for i in range(self.contract_selector.count())]:
                self.contract_selector.addItem(root)
        
        if strike not in self.trade_data[root]:
            self.trade_data[root][strike] = {'buys': 0, 'sells': 0, 'prices': []}
        
        if side == 'BUY':
            self.trade_data[root][strike]['buys'] += 1
        elif side == 'SELL':
            self.trade_data[root][strike]['sells'] += 1
            
        self.trade_data[root][strike]['prices'].append(price)
        self.update_chart()

    def create_tooltip(self, strike, data):
        avg_price = sum(data['prices']) / len(data['prices']) if data['prices'] else 0
        return (f"Strike: {strike}\n"
                f"Buys: {data['buys']}\n"
                f"Sells: {data['sells']}\n"
                f"Avg Price: ${avg_price:.2f}")

    def update_chart(self):
        # Clear everything
        self.chart.removeAllSeries()
        for axis in self.chart.axes():
            self.chart.removeAxis(axis)
        
        selected_root = self.contract_selector.currentText()
        if not selected_root or selected_root not in self.trade_data:
            return
            
        # Create bar sets for buys and sells
        buys = QBarSet("Buys")
        sells = QBarSet("Sells")
        
        # Set colors with transparency
        buys.setColor(Qt.green)
        sells.setColor(Qt.red)
        
        # Get strikes and sort them
        strikes = sorted(self.trade_data[selected_root].keys())
        strike_labels = [str(strike) for strike in strikes]
        
        # Add data to bar sets
        for strike in strikes:
            data = self.trade_data[selected_root][strike]
            buys.append(data['buys'])
            sells.append(data['sells'])
        
        # Create series
        series = QBarSeries()
        series.append(buys)
        series.append(sells)
        
        self.chart.addSeries(series)
        
        # Create axes
        axis_x = QBarCategoryAxis()
        axis_x.append(strike_labels)
        self.chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        self.chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)
        
        # Set labels
        axis_x.setTitleText("Strike Price")
        axis_y.setTitleText("Number of Trades")
        
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignBottom)

    def handle_hover(self, status, index, barset):
        if status:
            strike = self.chart.axes(Qt.Horizontal)[0].categories()[index]
            value = barset.at(index)
            tooltip = f"Strike: {strike}\n{barset.label()}: {value}"
            QToolTip.showText(QCursor.pos(), tooltip)

class StreamWindow(QMainWindow):
    def __init__(self, target_date=None, stream_manager=None):
        super().__init__()
        self.setWindowTitle("Option Trade Activity")
        self.setGeometry(100, 100, 1200, 800)
        self.stream_manager = stream_manager
        self.setup_ui()
        self.trade_count = 0
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        
        # Left panel for logs
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Status bar
        self.status_label = QLabel("Status: Starting...")
        self.stream_count_label = QLabel("Active Streams: 0")
        self.trade_count_label = QLabel("Trades: 0")
        
        status_layout = QVBoxLayout()
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.stream_count_label)
        status_layout.addWidget(self.trade_count_label)
        left_layout.addLayout(status_layout)
        
        # Trade log
        self.trade_log = QTextEdit()
        self.trade_log.setReadOnly(True)
        left_layout.addWidget(self.trade_log)
        
        layout.addWidget(left_panel)
        
        # Right panel for bar chart
        self.trade_chart = TradeActivityChart()
        layout.addWidget(self.trade_chart)
        
        # Set layout proportions
        layout.setStretch(0, 1)  # Left panel
        layout.setStretch(1, 1)  # Right panel
        
    def handle_trade(self, contract, trade, context):
        self.trade_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Format price and size
        price = float(trade['price'])
        size = int(trade['size'])
        side = trade['side']  # Get side directly from trade data
        
        # Create trade log entry with classification
        log_text = (f"{timestamp} - {contract['root']} "
                   f"{int(contract['strike'])/1000} {contract['right']} "
                   f"${price:.2f} x {size} "
                   f"[{side}]")
        
        # Add bid/ask context if available
        if context.get('bid') is not None and context.get('ask') is not None:
            log_text += f" (B: ${context['bid']:.2f} A: ${context['ask']:.2f})"
        
        log_text += "\n"
        
        self.trade_log.append(log_text)
        self.trade_count_label.setText(f"Trades: {self.trade_count}")
        
    def update_status(self, status):
        self.status_label.setText(f"Status: {status}")
        
    def update_stream_count(self, trade_count, quote_count):
        self.stream_count_label.setText(f"Active Streams: {trade_count}")
        self.trade_count_label.setText(f"Trades: {quote_count}")
        
    def log_error(self, error):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.trade_log.append(f"{timestamp} - ERROR: {error}\n")