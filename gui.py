import sys
from datetime import datetime, date
from PyQt5 import QtWidgets, QtCore, QtChart
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QComboBox, QCheckBox
from PyQt5.QtChart import QChart, QChartView, QBarSet, QBarSeries, QBarCategoryAxis, QValueAxis
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor, QCursor
from PyQt5.QtWidgets import QToolTip
import json
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from volatility_surface import BatchedVolatilitySurface
from option_chain import OptionChainService
import time

class VolatilitySurfaceChart(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        # Create horizontal layout for controls
        controls_layout = QHBoxLayout()
        
        # Add root selector
        self.root_selector = QComboBox()
        self.root_selector.currentTextChanged.connect(self.on_root_changed)
        controls_layout.addWidget(QLabel("Symbol:"))
        controls_layout.addWidget(self.root_selector)
        
        # Add expiry selector
        self.expiry_selector = QComboBox()
        self.expiry_selector.currentTextChanged.connect(self.on_expiry_changed)
        controls_layout.addWidget(QLabel("Expiry:"))
        controls_layout.addWidget(self.expiry_selector)
        
        # Add controls to main layout
        self.layout.addLayout(controls_layout)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        self.surface_ax = self.figure.add_subplot(111)
        self.layout.addWidget(self.canvas)
        
        # Initialize volatility surface with 1-second update interval
        self.vol_surface = BatchedVolatilitySurface(update_interval_ms=1000)
        self.vol_surface.surface_updated.connect(self.on_surface_updated)
        
        # Debug label
        self.debug_label = QLabel("No quotes processed yet")
        self.layout.addWidget(self.debug_label)
        
        # Setup update timer to limit GUI refresh rate
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_surface)
        self.update_timer.start(1000)  # Update GUI every second
        
        # Track pending updates
        self.pending_update = False
        self.last_update = time.time()

    def on_root_changed(self, root):
        """Handle root selection change"""
        if root:
            # Update expiry selector
            expiries = self.vol_surface.get_expiries(root)
            current_expiry = self.expiry_selector.currentText()
            
            self.expiry_selector.clear()
            self.expiry_selector.addItems(expiries)
            
            # Try to restore previous selection or select first available
            if current_expiry in expiries:
                self.expiry_selector.setCurrentText(current_expiry)
            elif expiries:
                self.expiry_selector.setCurrentIndex(0)
            
            self.request_update()
        
    def on_expiry_changed(self, expiry):
        """Handle expiry selection change"""
        if expiry:
            self.request_update()

    def on_surface_updated(self, root, expiry):
        """Handle surface update signal"""
        # Update root selector if needed
        if root not in [self.root_selector.itemText(i) for i in range(self.root_selector.count())]:
            self.root_selector.addItem(root)
            if not self.root_selector.currentText():
                self.root_selector.setCurrentText(root)
        
        # Update expiry selector
        if root == self.root_selector.currentText():
            expiries = self.vol_surface.get_expiries(root)
            current_expiry = self.expiry_selector.currentText()
            
            self.expiry_selector.clear()
            self.expiry_selector.addItems(expiries)
            
            # Try to restore previous selection or select first available
            if current_expiry in expiries:
                self.expiry_selector.setCurrentText(current_expiry)
            elif expiries:
                self.expiry_selector.setCurrentIndex(0)
            
            # Request update if this is the currently selected expiry
            if expiry == self.expiry_selector.currentText():
                self.request_update()

    def request_update(self):
        """Request a surface update"""
        self.pending_update = True

    def update_surface(self):
        """Update the volatility surface plot"""
        # Only update if there's a pending update and enough time has passed
        now = time.time()
        if not self.pending_update or (now - self.last_update) < 0.5:  # Limit to 2 updates per second
            return
            
        self.pending_update = False
        self.last_update = now
        
        try:
            self.surface_ax.clear()
            
            # Get current root and expiry from selectors
            root = self.root_selector.currentText()
            expiry = self.expiry_selector.currentText()
            if not root or not expiry:
                return
            
            # Get surface data for selected expiry
            surface_data = self.vol_surface.get_surface_data(root, expiry)
            if surface_data and surface_data[0]:  # Check if we have moneyness data
                moneyness, bid_vols, ask_vols, strikes = surface_data[0]
                
                # Plot vertical line at spot price (moneyness = 0)
                self.surface_ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Spot')
                
                # Plot puts and calls for all strikes
                # Plot puts
                self.surface_ax.scatter(
                    moneyness, 
                    bid_vols,
                    color='darkred', alpha=0.6, label='Puts Bid'
                )
                self.surface_ax.scatter(
                    moneyness, 
                    ask_vols,
                    color='red', alpha=0.6, label='Puts Ask'
                )
                
                # Plot calls
                self.surface_ax.scatter(
                    moneyness, 
                    bid_vols,
                    color='darkblue', alpha=0.6, label='Calls Bid'
                )
                self.surface_ax.scatter(
                    moneyness, 
                    ask_vols,
                    color='blue', alpha=0.6, label='Calls Ask'
                )
                
                # Add strike labels for reference (only every 5th point)
                for i, (m, v, k) in enumerate(zip(moneyness, bid_vols, strikes)):
                    if i % 5 == 0:  # Label every fifth point
                        self.surface_ax.annotate(f'{k:.1f}',
                                               (m, v),
                                               xytext=(0, 5),
                                               textcoords='offset points',
                                               ha='center',
                                               fontsize=8)
                
                self.surface_ax.set_xlabel('Log Moneyness')
                self.surface_ax.set_ylabel('Implied Volatility')
                self.surface_ax.set_title(f'Volatility Surface - {root} ({expiry})')
                self.surface_ax.grid(True)
                self.surface_ax.legend()
                
                # Set reasonable y-axis limits
                self.surface_ax.set_ylim(0, 1.0)  # Typical IV range
                
                # Update debug label with bid/ask info
                self.debug_label.setText(
                    f"Plotting {len(moneyness)} points | "
                    f"Moneyness: {min(moneyness):.3f} to {max(moneyness):.3f} | "
                    f"Bid IV: {min(bid_vols):.3%} to {max(bid_vols):.3%} | "
                    f"Ask IV: {min(ask_vols):.3%} to {max(ask_vols):.3%}"
                )
            else:
                self.debug_label.setText(f"No valid surface data for {root} ({expiry})")
                
            # Refresh the plot
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating surface: {str(e)}")

    def update_quote(self, contract, quote):
        """Process quote and update the volatility surface"""
        try:
            # Buffer the quote for batch processing
            self.vol_surface.update_quote(contract, quote)
        except Exception as e:
            print(f"Error processing quote: {str(e)}")

    def closeEvent(self, event):
        """Clean up resources on close"""
        if hasattr(self, 'vol_surface'):
            self.vol_surface.timer.stop()
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()
        super().closeEvent(event)

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
        
        # Right panel split between trade chart and volatility surface
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Trade activity chart
        self.trade_chart = TradeActivityChart()
        right_layout.addWidget(self.trade_chart)
        
        # Volatility surface chart
        self.vol_surface_chart = VolatilitySurfaceChart()
        right_layout.addWidget(self.vol_surface_chart)
        
        layout.addWidget(right_panel)
        
        # Set layout proportions
        layout.setStretch(0, 1)  # Left panel
        layout.setStretch(1, 2)  # Right panel
        
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
        
    def closeEvent(self, event):
        """Clean up resources on close"""
        if hasattr(self, 'vol_surface_chart'):
            self.vol_surface_chart.closeEvent(event)
        super().closeEvent(event)