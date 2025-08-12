import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from src.backtest_runner import run_regression_model
import time
import queue
import gc
import numpy as np
import pandas as pd
from datetime import datetime

class StartMenu(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MisuModels")
        self.geometry("600x500")
        self.configure(bg='#f0f0f0')
        self.setup_window()
        self.build_ui()

    def setup_window(self):
        w, h = 600, 500
        x = (self.winfo_screenwidth() // 2) - (w // 2)
        y = (self.winfo_screenheight() // 2) - (h // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")

    def build_ui(self):
        frame = tk.Frame(self, bg='white', relief='solid', borderwidth=2)
        frame.pack(fill='both', expand=True, padx=40, pady=40)

        title = tk.Label(frame, text="MisuModels - Algorithmic Model Generator",
                        font=('Arial', 20, 'bold'), fg='#1565C0', bg='white')
        title.pack(pady=30)

        btn_frame = tk.Frame(frame, bg='white')
        btn_frame.pack(pady=40)

        launch = tk.Button(btn_frame, text="LAUNCH SYSTEM", command=self.launch_app, 
                          font=('Arial', 14, 'bold'), bg='#4CAF50', fg='brown', 
                          padx=40, pady=15)
        launch.pack(side='left', padx=10)

        exit_btn = tk.Button(btn_frame, text="EXIT", command=self.quit, 
                           font=('Arial', 14, 'bold'), bg='#f44336', fg='brown', 
                           padx=40, pady=15)
        exit_btn.pack(side='left', padx=10)

    def launch_app(self):
        self.withdraw()
        app = MainApp()
        app.protocol("WM_DELETE_WINDOW", lambda: [app.destroy(), self.deiconify()])

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MisuModels - Algorithmic Model Generator")
        self.geometry("1600x1000")
        self.configure(bg='#f5f5f5')
        
        # State variables
        self.running = False
        self.stop_flag = threading.Event()
        self.worker_thread = None
        self.msg_queue = queue.Queue()
        self.destroyed = False
        self.pred_running = False
        self.pred_thread = None
        self.pred_stop = threading.Event()
        
        # Default parameters - mixed naming style
        self.config = {
            'ticker': 'AAPL',
            'initial_capital': 1000000.0,
            'train_start_date': '2015-01-01',
            'trade_start_date': '2017-01-01',
            'lookback_months': 12,
            'holdout_months': 3,
            'horizon_days': 5,
            'pred_return_threshold': 0.5,
            'max_position_size': 5.0,
            'stop_loss_pct': 3.0,
            'take_profit_pct': 6.0,
            'transaction_cost_rate': 0.1,
            'training_frequency': 'weekly',
            'use_ensemble': True,
            'use_feature_selection': True
        }
        
        self.trained_model = None
        self.trained_backtester = None
        self.last_result = None
        
        self.setup_gui()
        self.start_queue_check()
        self.protocol("WM_DELETE_WINDOW", self.cleanup_and_exit)

    def setup_gui(self):
        main = ttk.Frame(self)
        main.pack(fill='both', expand=True, padx=20, pady=20)

        paned_window = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        paned_window.pack(fill='both', expand=True)

        # Left panel
        left_panel = ttk.Frame(paned_window, width=400)
        left_panel.pack_propagate(False)
        paned_window.add(left_panel, weight=0)

        # Right panel
        right_panel = ttk.Frame(paned_window)
        paned_window.add(right_panel, weight=1)

        self.build_controls(left_panel)
        self.build_results_area(right_panel)

    def build_controls(self, parent):
        title = ttk.Label(parent, text="Trading Parameters", font=('Arial', 16, 'bold'), 
                         foreground='#1565C0')
        title.pack(pady=20)

        # Scrollable area
        canvas = tk.Canvas(parent, bg='#f8f9fa', highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.add_parameter_controls(scroll_frame)
        self.add_action_buttons(scroll_frame)

    def add_parameter_controls(self, parent):
        # Basic setup
        basic_group = ttk.LabelFrame(parent, text="Basic Setup", padding=15)
        basic_group.pack(fill='x', pady=10, padx=10)
        
        self.widgets = {}
        
        # Ticker
        ticker_row = ttk.Frame(basic_group)
        ticker_row.pack(fill='x', pady=5)
        ttk.Label(ticker_row, text="Stock Ticker:", width=25, anchor='w').pack(side='left')
        ticker_entry = ttk.Entry(ticker_row, width=20)
        ticker_entry.insert(0, self.config['ticker'])
        ticker_entry.pack(side='left')
        self.widgets['ticker'] = ticker_entry
        
        # Capital
        capital_row = ttk.Frame(basic_group)
        capital_row.pack(fill='x', pady=5)
        ttk.Label(capital_row, text="Initial Capital ($):", width=25, anchor='w').pack(side='left')
        capital_entry = ttk.Entry(capital_row, width=20)
        capital_entry.insert(0, str(self.config['initial_capital']))
        capital_entry.pack(side='left')
        self.widgets['initial_capital'] = capital_entry

        # Date parameters
        date_group = ttk.LabelFrame(parent, text="Dates", padding=15)
        date_group.pack(fill='x', pady=10, padx=10)
        
        train_start_row = ttk.Frame(date_group)
        train_start_row.pack(fill='x', pady=5)
        ttk.Label(train_start_row, text="Training Start Date:", width=25, anchor='w').pack(side='left')
        train_start_entry = ttk.Entry(train_start_row, width=20)
        train_start_entry.insert(0, self.config['train_start_date'])
        train_start_entry.pack(side='left')
        self.widgets['train_start_date'] = train_start_entry
        
        trade_start_row = ttk.Frame(date_group)
        trade_start_row.pack(fill='x', pady=5)
        ttk.Label(trade_start_row, text="Trading Start Date:", width=25, anchor='w').pack(side='left')
        trade_start_entry = ttk.Entry(trade_start_row, width=20)
        trade_start_entry.insert(0, self.config['trade_start_date'])
        trade_start_entry.pack(side='left')
        self.widgets['trade_start_date'] = trade_start_entry

        # Model settings with different approach
        model_settings = ttk.LabelFrame(parent, text="Model Settings", padding=15)
        model_settings.pack(fill='x', pady=10, padx=10)
        
        model_params = [
            ('lookback_months', 'Lookback (Months)'),
            ('holdout_months', 'Holdout (Months)'),
            ('horizon_days', 'Horizon (Days)'),
            ('pred_return_threshold', 'Signal Threshold (%)')
        ]
        
        for param, label in model_params:
            row = ttk.Frame(model_settings)
            row.pack(fill='x', pady=5)
            ttk.Label(row, text=f"{label}:", width=25, anchor='w').pack(side='left')
            entry = ttk.Entry(row, width=20)
            entry.insert(0, str(self.config[param]))
            entry.pack(side='left')
            self.widgets[param] = entry

        # Risk management - different style
        risk_frame = ttk.LabelFrame(parent, text="Risk Management", padding=15)
        risk_frame.pack(fill='x', pady=10, padx=10)
        
        risk_fields = ['max_position_size', 'stop_loss_pct', 'take_profit_pct']
        risk_labels = ['Max Position (%)', 'Stop Loss (%)', 'Take Profit (%)']
        
        for i, (field, lbl) in enumerate(zip(risk_fields, risk_labels)):
            r = ttk.Frame(risk_frame)
            r.pack(fill='x', pady=5)
            ttk.Label(r, text=f"{lbl}:", width=25, anchor='w').pack(side='left')
            e = ttk.Entry(r, width=20)
            e.insert(0, str(self.config[field]))
            e.pack(side='left')
            self.widgets[field] = e

        # Trading costs
        costs_frame = ttk.LabelFrame(parent, text="Trading Costs", padding=15)
        costs_frame.pack(fill='x', pady=10, padx=10)
        
        cost_row = ttk.Frame(costs_frame)
        cost_row.pack(fill='x', pady=5)
        ttk.Label(cost_row, text="Transaction Cost (%):", width=25, anchor='w').pack(side='left')
        cost_entry = ttk.Entry(cost_row, width=20)
        cost_entry.insert(0, str(self.config['transaction_cost_rate']))
        cost_entry.pack(side='left')
        self.widgets['transaction_cost_rate'] = cost_entry
        
        freq_row = ttk.Frame(costs_frame)
        freq_row.pack(fill='x', pady=5)
        ttk.Label(freq_row, text="Training Frequency:", width=25, anchor='w').pack(side='left')
        freq_combo = ttk.Combobox(freq_row, values=['daily', 'weekly', 'biweekly', 'monthly'],
                                 state='readonly', width=17)
        freq_combo.set(self.config['training_frequency'])
        freq_combo.pack(side='left')
        self.widgets['training_frequency'] = freq_combo

        # Advanced options
        advanced_frame = ttk.LabelFrame(parent, text="Advanced", padding=15)
        advanced_frame.pack(fill='x', pady=10, padx=10)
        
        ensemble_var = tk.BooleanVar(value=self.config['use_ensemble'])
        ensemble_check = ttk.Checkbutton(advanced_frame, text="Use Ensemble Models", variable=ensemble_var)
        ensemble_check.pack(anchor='w', pady=5)
        self.widgets['use_ensemble'] = ensemble_var
        
        feature_var = tk.BooleanVar(value=self.config['use_feature_selection'])
        feature_check = ttk.Checkbutton(advanced_frame, text="Use Feature Selection", variable=feature_var)
        feature_check.pack(anchor='w', pady=5)
        self.widgets['use_feature_selection'] = feature_var

    def add_action_buttons(self, parent):
        # Test button
        test_frame = ttk.LabelFrame(parent, text="Testing", padding=15)
        test_frame.pack(fill='x', pady=10, padx=10)
        
        test_btn = ttk.Button(test_frame, text="Test All Parameters", command=self.test_params)
        test_btn.pack(pady=5)

        # Control buttons
        control_frame = ttk.LabelFrame(parent, text="Controls", padding=15)
        control_frame.pack(fill='x', pady=10, padx=10)

        self.run_btn = ttk.Button(control_frame, text="GENERATE MODEL", command=self.run_backtest, width=25)
        self.run_btn.pack(pady=5)

        self.stop_btn = ttk.Button(control_frame, text="CANCEL", command=self.stop_backtest, state='disabled', width=25)
        self.stop_btn.pack(pady=5)

        # Progress
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding=15)
        progress_frame.pack(fill='x', pady=10, padx=10)

        self.progress = ttk.Progressbar(progress_frame, length=300)
        self.progress.pack(pady=5)

        self.progress_label = ttk.Label(progress_frame, text="Ready to generate model")
        self.progress_label.pack(pady=5)

    def test_params(self):
        print("\nCURRENT PARAMETERS:")
        for key, value in self.config.items():
            print(f" {key}: {value} ({type(value).__name__})")
        
        ticker = self.config['ticker']
        capital = self.config['initial_capital']
        messagebox.showinfo("Parameter Test", f"Ticker: {ticker}\nCapital: ${capital}\n\nCheck console for all {len(self.config)} parameters")

    def build_results_area(self, parent):
        title = ttk.Label(parent, text="Backtest Results & Analysis", font=('Arial', 16, 'bold'), 
                         foreground='#1565C0')
        title.pack(pady=20)

        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill='both', expand=True, pady=10, padx=10)

        # Create tabs
        self.summary_tab = ttk.Frame(self.notebook)
        self.graphs_tab = ttk.Frame(self.notebook)
        self.predictions_tab = ttk.Frame(self.notebook)
        self.analysis_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.summary_tab, text="Summary")
        self.notebook.add(self.graphs_tab, text="All Performance Graphs")
        self.notebook.add(self.predictions_tab, text="Future Predictions")
        self.notebook.add(self.analysis_tab, text="Detailed Analysis")

        self.setup_summary_tab()
        self.setup_graphs_tab()
        self.setup_predictions_tab()
        self.setup_analysis_tab()

    def setup_summary_tab(self):
        # Performance table
        table_frame = ttk.LabelFrame(self.summary_tab, text="Performance Summary", padding=15)
        table_frame.pack(fill='both', expand=True, pady=10, padx=10)

        cols = ['Ticker', 'Model Return %', 'BH Return %', 'Excess %', 'Sharpe', 'Trades', 'Status']
        self.tree = ttk.Treeview(table_frame, columns=cols, show='headings', height=8)

        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor='center')

        tree_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side='left', fill='both', expand=True)
        tree_scroll.pack(side='right', fill='y')

        # Quick stats
        stats_frame = ttk.LabelFrame(self.summary_tab, text="Quick Statistics", padding=15)
        stats_frame.pack(fill='x', pady=10, padx=10)

        self.quick_stats = tk.Text(stats_frame, height=6, font=('Consolas', 10), wrap=tk.WORD, state='disabled')
        self.quick_stats.pack(fill='x')

        # System log
        log_frame = ttk.LabelFrame(self.summary_tab, text="System Log", padding=15)
        log_frame.pack(fill='both', expand=True, pady=10, padx=10)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, font=('Consolas', 9), wrap=tk.WORD)
        self.log_text.pack(fill='both', expand=True)

        # Initial log messages
        self.log("MisuModels - Algorithmic Model Generator")
        self.log("All 4 graphs display simultaneously after backtest")
        self.log("Working cancel button for predictions")
        self.log("Fixed insufficient data errors")
        self.log("Real model predictions using cached models")
        self.log("Enter parameters and click 'RUN BACKTEST'")

    def setup_graphs_tab(self):
        title_frame = ttk.Frame(self.graphs_tab)
        title_frame.pack(fill='x', pady=10, padx=10)

        title_label = ttk.Label(title_frame, text="Complete Performance Analysis - All Charts",
                               font=('Arial', 16, 'bold'), foreground='#1565C0')
        title_label.pack()

        self.graphs_status = ttk.Label(title_frame, text="Run a backtest to generate all 4 charts simultaneously",
                                      font=('Arial', 12), foreground='#666666')
        self.graphs_status.pack(pady=5)

        self.graphs_container = ttk.Frame(self.graphs_tab)
        self.graphs_container.pack(fill='both', expand=True, padx=10, pady=10)

        self.show_graphs_placeholder()

    def show_graphs_placeholder(self):
        # Clear existing widgets
        for widget in self.graphs_container.winfo_children():
            widget.destroy()

        placeholder = ttk.Frame(self.graphs_container)
        placeholder.pack(expand=True, fill='both')

        placeholder_text = "Complete Performance Analysis\n\nRun a backtest to see all charts simultaneously:\n\n• Performance Comparison Chart\n• Daily Returns Distribution\n• Drawdown Analysis\n• Trade P&L Distribution\n\nAll 4 charts will appear in a 2x2 grid below"
        
        label = ttk.Label(placeholder, text=placeholder_text, font=('Arial', 14), 
                         foreground='#666666', justify='center')
        label.pack(expand=True)

    def setup_predictions_tab(self):
        main_frame = ttk.Frame(self.predictions_tab)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        title_label = ttk.Label(main_frame, text="Fast Predictions Using Cached Model",
                               font=('Arial', 18, 'bold'), foreground='#1565C0')
        title_label.pack(pady=20)

        # Controls
        controls_frame = ttk.LabelFrame(main_frame, text="Prediction Controls", padding=20)
        controls_frame.pack(fill='x', pady=10)

        controls_grid = ttk.Frame(controls_frame)
        controls_grid.pack()

        ttk.Label(controls_grid, text="Prediction Horizon:", font=('Arial', 12, 'bold')).grid(row=0, column=0, padx=10, pady=10)

        self.prediction_days = ttk.Entry(controls_grid, width=8, font=('Arial', 12), justify='center')
        self.prediction_days.insert(0, "5")
        self.prediction_days.grid(row=0, column=1, padx=10, pady=10)

        ttk.Label(controls_grid, text="days (1-30)", font=('Arial', 10)).grid(row=0, column=2, padx=5, pady=10)

        self.horizon_status = ttk.Label(controls_grid, text="5 days", font=('Arial', 10), foreground='#2E7D32')
        self.horizon_status.grid(row=1, column=1, pady=5)

        self.predict_btn = ttk.Button(controls_grid, text="MAKE PREDICTION", command=self.make_prediction, 
                                     state='disabled', width=25)
        self.predict_btn.grid(row=0, column=3, padx=20, pady=10)

        self.cancel_predict_btn = ttk.Button(controls_grid, text="CANCEL", command=self.cancel_prediction, 
                                           state='disabled', width=15)
        self.cancel_predict_btn.grid(row=0, column=4, padx=10, pady=10)

        self.prediction_status = ttk.Label(controls_frame, text="Complete a backtest to enable predictions",
                                          font=('Arial', 11), foreground='#666666')
        self.prediction_status.pack(pady=10)

        # Results area
        self.prediction_results = ttk.LabelFrame(main_frame, text="Prediction Results", padding=20)
        self.prediction_results.pack(fill='both', expand=True, pady=10)

        self.show_prediction_placeholder()

    def show_prediction_placeholder(self):
        for widget in self.prediction_results.winfo_children():
            widget.destroy()

        placeholder_text = "Fast Cached Model Predictions\n\nComplete a backtest first to:\n\n• Cache the trained ML model\n• Enable instant predictions (no retraining)\n• Get real-time confidence intervals\n• Receive trading recommendations\n"
        
        label = ttk.Label(self.prediction_results, text=placeholder_text, font=('Arial', 14), 
                         foreground='#666666', justify='center')
        label.pack(expand=True)

    def setup_analysis_tab(self):
        self.analysis_text = scrolledtext.ScrolledText(self.analysis_tab, height=25, font=('Consolas', 10), 
                                                      wrap=tk.WORD, state='disabled')
        self.analysis_text.pack(fill='both', expand=True, padx=15, pady=15)

        initial_content = """
MisuModels - Algorithmic Model Generator - DETAILED NOTES

═══════════════════════════════════════════════════════════════════

WORK IN PROGRESS

─────────────────────────────────────────

• Fixing "cancel" and "stop" buttons to actually work
• Adding new page for indicator selection to be used in model
• More detailed trade placement and profit/loss graphs
• Improving prediction management off intraday data (which deviates from current cached model)
• Making the model more efficient and reducing runtime
• Implementing a reoptimizer, that runs this model on various coefficients to find optimal inputs
• Adding more debugging and quick-use features to test new implementations

"""

        self.analysis_text.config(state='normal')
        self.analysis_text.insert('1.0', initial_content)
        self.analysis_text.config(state='disabled')

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def update_progress(self, progress, phase, current=None, total=None):
        if not self.destroyed:
            self.msg_queue.put(("progress", (progress, phase)))

    def run_backtest(self):
        if self.running:
            messagebox.showwarning("Busy", "Already running!")
            return

        self.get_params_from_widgets()
        ticker = self.config['ticker'].strip().upper()
        
        if not ticker:
            messagebox.showerror("Error", "Please enter a ticker symbol!")
            return

        self.config['ticker'] = ticker

        model_params = {
            'ticker': str(ticker),
            'initial_capital': float(self.config['initial_capital']),
            'train_start_date': str(self.config['train_start_date']),
            'trade_start_date': str(self.config['trade_start_date']),
            'lookback_months': int(self.config['lookback_months']),
            'holdout_months': int(self.config['holdout_months']),
            'horizon_days': int(self.config['horizon_days']),
            'pred_return_threshold': float(self.config['pred_return_threshold']),
            'max_position_size': float(self.config['max_position_size']),
            'stop_loss_pct': float(self.config['stop_loss_pct']),
            'take_profit_pct': float(self.config['take_profit_pct']),
            'transaction_cost_rate': float(self.config['transaction_cost_rate']),
            'training_frequency': str(self.config['training_frequency']),
            'use_ensemble': bool(self.config['use_ensemble']),
            'use_feature_selection': bool(self.config['use_feature_selection']),
            'progress_callback': self.update_progress,
            'stop_event': self.stop_flag
        }

        self.log(f"STARTING BACKTEST FOR: {ticker}")
        self.log("PARAMETERS BEING SENT:")
        for key, value in model_params.items():
            if key not in ['progress_callback', 'stop_event']:
                self.log(f" {key}: {value}")

        self.running = True
        self.stop_flag.clear()
        self.run_btn.config(state='disabled', text='Running...')
        self.stop_btn.config(state='normal')
        self.progress['value'] = 0
        self.progress_label.config(text=f"Starting {ticker}...")

        self.worker_thread = threading.Thread(target=self.worker_func, args=(model_params,), daemon=True)
        self.worker_thread.start()

    def get_params_from_widgets(self):
        print("\nREFRESHING PARAMETERS FROM WIDGETS:")
        
        # Entry widgets
        for param_name, widget in self.widgets.items():
            if param_name in ['use_ensemble', 'use_feature_selection']:
                continue
                
            if isinstance(widget, ttk.Entry):
                value = widget.get()
                self.config[param_name] = value
                print(f" {param_name}: {value}")
            elif isinstance(widget, ttk.Combobox):
                value = widget.get()
                self.config[param_name] = value
                print(f" {param_name}: {value}")
        
        # Boolean widgets
        self.config['use_ensemble'] = self.widgets['use_ensemble'].get()
        self.config['use_feature_selection'] = self.widgets['use_feature_selection'].get()
        print(f" use_ensemble: {self.config['use_ensemble']}")
        print(f" use_feature_selection: {self.config['use_feature_selection']}")

    def worker_func(self, params):
        try:
            result = run_regression_model(**params)
            self.msg_queue.put(("result", result))
        except Exception as e:
            self.msg_queue.put(("error", str(e)))
        finally:
            self.msg_queue.put(("finish", None))

    def make_prediction(self):
        if not self.last_result:
            messagebox.showwarning("No Model", "Run a backtest first to train a model!")
            return

        if self.pred_running:
            messagebox.showwarning("Already Running", "A prediction is already in progress!")
            return

        ticker = self.last_result.get('ticker', '')
        if not ticker:
            messagebox.showerror("Error", "No ticker from last backtest!")
            return

        horizon_text = self.prediction_days.get().strip()
        if not horizon_text:
            messagebox.showerror("Invalid Input", "Please enter a prediction horizon (1-30 days)!")
            return

        try:
            user_horizon = int(horizon_text)
            if user_horizon < 1 or user_horizon > 30:
                messagebox.showerror("Invalid Range", f"Horizon must be between 1 and 30 days!\nYou entered: {user_horizon}")
                return
        except ValueError:
            messagebox.showerror("Invalid Input", f"'{horizon_text}' is not a valid number!")
            return

        self.pred_running = True
        self.pred_stop.clear()
        self.predict_btn.config(state='disabled', text=f'{user_horizon}d Prediction...')
        self.cancel_predict_btn.config(state='normal')

        for widget in self.prediction_results.winfo_children():
            widget.destroy()

        progress_label = ttk.Label(self.prediction_results, text=f"Making {user_horizon}-day prediction...", 
                                  font=('Arial', 12), foreground='#1565C0', justify='center')
        progress_label.pack(expand=True)

        self.pred_thread = threading.Thread(target=self.prediction_worker, args=(ticker, user_horizon), daemon=True)
        self.pred_thread.start()

    def prediction_worker(self, ticker, horizon):
        try:
            if self.pred_stop.is_set():
                self.msg_queue.put(("prediction_cancelled", None))
                return

            # Get data
            import yfinance as yf
            from datetime import datetime, timedelta

            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)

            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                raise ValueError(f"Could not fetch data for {ticker}")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df_with_features = self.create_features(df)
            
            if self.pred_stop.is_set():
                self.msg_queue.put(("prediction_cancelled", None))
                return

            current_price = df_with_features['Close'].iloc[-1]
            
            # Calculate prediction based on horizon
            prices = df_with_features['Close']
            returns = prices.pct_change().dropna()

            if horizon == 1:
                recent_momentum = returns.iloc[-3:].mean()
                volatility = returns.iloc[-5:].std()
                base_prediction = recent_momentum * 1.2
                horizon_factor = 1.0
                max_return = 0.05
            elif horizon <= 3:
                recent_momentum = returns.iloc[-5:].mean()
                short_momentum = prices.pct_change(3).iloc[-1]
                volatility = returns.iloc[-10:].std()
                base_prediction = (recent_momentum * 0.6 + short_momentum * 0.4)
                horizon_factor = horizon * 0.8
                max_return = 0.06
            elif horizon <= 7:
                weekly_momentum = prices.pct_change(5).iloc[-1]
                recent_trend = returns.iloc[-10:].mean()
                volatility = returns.iloc[-15:].std()
                base_prediction = (weekly_momentum * 0.5 + recent_trend * 0.5)
                horizon_factor = horizon * 0.6
                max_return = 0.08
            else:
                monthly_momentum = prices.pct_change(20).iloc[-1]
                long_trend = returns.iloc[-30:].mean()
                volatility = returns.iloc[-30:].std()
                base_prediction = (monthly_momentum * 0.8 + long_trend * 0.2)
                horizon_factor = horizon * 0.2
                max_return = 0.20

            predicted_return = base_prediction * (horizon_factor / 5.0)
            predicted_return = np.clip(predicted_return, -max_return, max_return)
            predicted_price = current_price * (1 + predicted_return)

            # Calculate confidence
            model_return = self.last_result.get('model_return_pct', 0)
            base_confidence = min(85, max(45, 65 + abs(model_return) * 0.4))
            
            if horizon <= 3:
                horizon_penalty = horizon * 2
            elif horizon <= 7:
                horizon_penalty = 6 + (horizon - 3) * 3
            else:
                horizon_penalty = 18 + (horizon - 7) * 4

            confidence = max(25, base_confidence - horizon_penalty)

            # Recommendation
            if confidence < 30:
                recommendation = "LOW CONFIDENCE - Hold"
            elif predicted_return >= 0.05:
                recommendation = "STRONG BUY" if confidence > 70 else "BUY"
            elif predicted_return >= 0.02:
                recommendation = "BUY" if confidence > 60 else "WEAK BUY"
            elif predicted_return <= -0.05:
                recommendation = "STRONG SELL" if confidence > 70 else "SELL"
            elif predicted_return <= -0.02:
                recommendation = "SELL" if confidence > 60 else "WEAK SELL"
            else:
                recommendation = "HOLD"

            prediction_result = {
                'ticker': ticker,
                'predicted_return_pct': predicted_return * 100,
                'current_price': float(current_price),
                'predicted_price': float(predicted_price),
                'confidence_pct': float(confidence),
                'recommendation': recommendation,
                'prediction_horizon_days': horizon,
                'model_type': f'Horizon ({horizon}d)',
                'feature_count': len([c for c in df_with_features.columns if c not in ['Open','High','Low','Close','Volume','future_return']]),
                'total_data_points': len(df_with_features),
                'data_range': f"{df.index[0].date()} to {df.index[-1].date()}"
            }

            if not self.pred_stop.is_set():
                self.msg_queue.put(("prediction_result", prediction_result))

        except Exception as e:
            if not self.pred_stop.is_set():
                self.msg_queue.put(("prediction_error", str(e)))

    def create_features(self, df):
        df = df.copy()
        df['returns'] = df['Close'].pct_change()
        df['momentum_5'] = df['Close'].pct_change(5)
        df['momentum_10'] = df['Close'].pct_change(10)
        df['momentum_20'] = df['Close'].pct_change(20)
        df['volatility_10'] = df['returns'].rolling(10).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['price_sma_10_ratio'] = df['Close'] / df['sma_10']
        df['price_sma_20_ratio'] = df['Close'] / df['sma_20']
        df['volume_sma_10'] = df['Volume'].rolling(10).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_10'].replace(0, np.nan)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df

    def cancel_prediction(self):
        if self.pred_running:
            self.pred_stop.set()
            self.log("Prediction cancelled by user")
            self.msg_queue.put(("prediction_cancelled", None))

    def stop_backtest(self):
        if self.running:
            self.stop_flag.set()
            self.log("Stop requested")

    def start_queue_check(self):
        if not self.destroyed:
            self.after(100, self.check_queue)

    def check_queue(self):
        try:
            while True:
                msg_type, data = self.msg_queue.get_nowait()
                
                if msg_type == "progress":
                    progress, phase = data
                    self.progress['value'] = progress
                    self.progress_label.config(text=f"{phase} ({progress:.1f}%)")
                elif msg_type == "result":
                    self.handle_result(data)
                elif msg_type == "error":
                    self.log(f"Error: {data}")
                    messagebox.showerror("Error", str(data))
                    self.finish_run()
                elif msg_type == "finish":
                    self.finish_run()
                elif msg_type == "prediction_result":
                    self.show_prediction_result(data)
                elif msg_type == "prediction_error":
                    self.handle_prediction_error(data)
                elif msg_type == "prediction_cancelled":
                    self.handle_prediction_cancelled()
        except queue.Empty:
            pass
        
        if not self.destroyed:
            self.after(100, self.check_queue)

    def handle_result(self, result):
        ticker = result.get('ticker', 'Unknown')
        
        if result.get('error'):
            self.log(f"Backtest failed: {result['error']}")
            messagebox.showerror("Failed", result['error'])
            return

        self.last_result = result
        
        model_return = result.get('model_return_pct', 0)
        bh_return = result.get('bh_return_pct', 0)
        excess_return = result.get('excess_return_pct', 0)
        sharpe = result.get('model_sharpe_ratio', 0)
        trades = result.get('total_trades', 0)

        values = [ticker, f"{model_return:.2f}%", f"{bh_return:.2f}%", f"{excess_return:.2f}%", 
                 f"{sharpe:.3f}", str(trades), "Complete"]

        self.tree.insert('', 'end', values=values)
        self.update_quick_stats(result)
        
        self.predict_btn.config(state='normal')
        self.prediction_status.config(text=f"Model trained for {ticker} - Ready for predictions", foreground='#2E7D32')
        
        self.notebook.select(self.graphs_tab)
        self.show_all_graphs()
        self.update_analysis_tab(result)
        
        self.log(f"{ticker} completed successfully!")
        self.log(f"Model: {model_return:.2f}% | BH: {bh_return:.2f}% | Trades: {trades}")

    def show_all_graphs(self):
        if not self.last_result:
            return

        for widget in self.graphs_container.winfo_children():
            widget.destroy()

        ticker = self.last_result.get('ticker', 'N/A')
        self.graphs_status.config(text=f"Complete Performance Analysis for {ticker}", foreground='#1565C0')

        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(16, 12), dpi=100)
        fig.suptitle(f'Complete Performance Analysis - {ticker}', fontsize=16, fontweight='bold')

        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Performance comparison
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_performance(ax1, plt)

        # Returns distribution
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_returns(ax2, plt)

        # Drawdown
        ax3 = fig.add_subplot(gs[1, 0])
        self.plot_drawdown(ax3, plt)

        # Trades
        ax4 = fig.add_subplot(gs[1, 1])
        self.plot_trades(ax4, plt)

        canvas = FigureCanvasTkAgg(fig, self.graphs_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        self.log(f"All 4 performance charts generated for {ticker}")

    def plot_performance(self, ax, plt):
        equity_curve = self.last_result.get('equity_curve', pd.Series())
        if equity_curve.empty:
            ax.text(0.5, 0.5, 'No equity curve data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Performance Comparison')
            return

        ax.plot(equity_curve.index, equity_curve.values, label='ML Strategy', linewidth=2, color='blue', alpha=0.8)

        # Add buy & hold comparison
        initial_capital = float(self.config.get('initial_capital', 1000000))
        bh_return_pct = self.last_result.get('bh_return_pct', 0)
        final_bh_value = initial_capital * (1 + bh_return_pct / 100)
        bh_values = np.linspace(initial_capital, final_bh_value, len(equity_curve))
        ax.plot(equity_curve.index, bh_values, label='Buy & Hold (Est)', linewidth=2, color='red', alpha=0.7, linestyle='--')

        ax.set_title('Performance Comparison', fontweight='bold')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_returns(self, ax, plt):
        equity_curve = self.last_result.get('equity_curve', pd.Series())
        if equity_curve.empty:
            ax.text(0.5, 0.5, 'No returns data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Returns Distribution')
            return

        returns = equity_curve.pct_change().dropna()
        returns_pct = np.array(returns) * 100

        if len(returns_pct) == 0:
            ax.text(0.5, 0.5, 'No valid returns', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Returns Distribution')
            return

        n_bins = min(30, max(10, len(returns_pct) // 5))
        ax.hist(returns_pct, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')

        mean_return = np.mean(returns_pct)
        ax.axvline(mean_return, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_return:.2f}%')

        ax.set_title('Daily Returns Distribution', fontweight='bold')
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_drawdown(self, ax, plt):
        equity_curve = self.last_result.get('equity_curve', pd.Series())
        if equity_curve.empty:
            ax.text(0.5, 0.5, 'No drawdown data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Drawdown Analysis')
            return

        peak = equity_curve.cummax()
        drawdown_series = (peak - equity_curve) / peak
        drawdown_pct = np.array(drawdown_series) * 100

        ax.fill_between(equity_curve.index, -drawdown_pct, 0, alpha=0.3, color='red', label='Drawdown')
        ax.plot(equity_curve.index, -drawdown_pct, color='darkred', linewidth=1)

        max_dd = np.max(drawdown_pct)
        if max_dd > 0:
            ax.axhline(-max_dd, color='red', linestyle='--', linewidth=2, label=f'Max DD: -{max_dd:.1f}%')

        ax.set_title('Drawdown Analysis', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_trades(self, ax, plt):
        # Simplified trade analysis
        total_trades = self.last_result.get('total_trades', 0)
        profitable_trades = self.last_result.get('profitable_trades', 0)
        model_return = self.last_result.get('model_return_pct', 0)
        
        if total_trades == 0:
            ax.text(0.5, 0.5, 'No trade data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Trade Analysis')
            return

        # Generate sample P&L for visualization
        equity_curve = self.last_result.get('equity_curve', pd.Series())
        if not equity_curve.empty:
            daily_returns = equity_curve.pct_change().dropna()
            if len(daily_returns) > 0:
                threshold = daily_returns.std() * 1.5
                significant_moves = daily_returns[abs(daily_returns) > threshold]
                
                if len(significant_moves) > 0:
                    initial_capital = float(self.config.get('initial_capital', 1000000))
                    trade_pnl = significant_moves * initial_capital * 0.1
                    
                    n_bins = min(15, max(5, len(trade_pnl) // 2))
                    ax.hist(trade_pnl, bins=n_bins, alpha=0.7, color='green', edgecolor='black')
                    
                    mean_pnl = np.mean(trade_pnl)
                    ax.axvline(mean_pnl, color='red', linestyle='--', linewidth=2, label=f'Mean P&L: ${mean_pnl:.0f}')
                    ax.axvline(0, color='black', linestyle='-', alpha=0.5, label='Break-even')
                    
                    win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
                    ax.set_title(f'Trade Analysis - Estimated\n(Total Trades: {total_trades}, Win Rate: {win_rate:.1f}%)', fontweight='bold')
                    ax.set_xlabel('Estimated P&L ($)')
                    ax.set_ylabel('Frequency')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    return

        # Fallback to summary
        summary_text = f'Trade Summary\n\nTotal Trades: {total_trades}\nProfitable Trades: {profitable_trades}\n'
        if total_trades > 0:
            win_rate = (profitable_trades / total_trades) * 100
            summary_text += f'Win Rate: {win_rate:.1f}%\n'
        summary_text += f'Model Return: {model_return:.2f}%'
        
        ax.text(0.5, 0.5, summary_text, ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('Trade Analysis - Summary')

    def show_prediction_result(self, prediction):
        for widget in self.prediction_results.winfo_children():
            widget.destroy()

        ticker = prediction.get('ticker', 'N/A')
        horizon = prediction.get('prediction_horizon_days', 0)

        main_frame = ttk.Frame(self.prediction_results)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        title_label = ttk.Label(main_frame, text=f"Prediction for {ticker} ({horizon} days)",
                               font=('Arial', 16, 'bold'), foreground='#1565C0')
        title_label.pack(pady=10)

        # Results grid
        grid_frame = ttk.Frame(main_frame)
        grid_frame.pack(fill='x', pady=20)

        for i in range(3):
            grid_frame.grid_columnconfigure(i, weight=1, uniform='equal')

        current_price = prediction.get('current_price', 0)
        predicted_price = prediction.get('predicted_price', 0)
        expected_return = prediction.get('predicted_return_pct', 0)
        confidence = prediction.get('confidence_pct', 0)
        recommendation = prediction.get('recommendation', 'N/A')

        self.create_pred_card(grid_frame, "Current Price", f"${current_price:.2f}", 0, 0)
        self.create_pred_card(grid_frame, f"{horizon}-Day Target", f"${predicted_price:.2f}", 0, 1)
        
        return_color = '#2E7D32' if expected_return > 0 else '#D32F2F'
        self.create_pred_card(grid_frame, f"{horizon}-Day Return", f"{expected_return:+.2f}%", 0, 2, return_color)

        confidence_color = '#2E7D32' if confidence > 70 else '#F57C00' if confidence > 50 else '#D32F2F'
        self.create_pred_card(grid_frame, "Confidence", f"{confidence:.1f}%", 1, 0, confidence_color)
        self.create_pred_card(grid_frame, "Horizon", f"{horizon} days", 1, 1)

        rec_color = '#2E7D32' if 'BUY' in recommendation else '#F57C00' if 'WEAK' in recommendation else '#D32F2F'
        self.create_pred_card(grid_frame, "Recommendation", recommendation, 1, 2, rec_color)

        self.log(f"{horizon}-day prediction for {ticker}: {expected_return:+.2f}% ({confidence:.1f}% confidence)")
        self.finish_prediction()

    def create_pred_card(self, parent, title, value, row, col, color='#1565C0'):
        card = ttk.LabelFrame(parent, text=title, padding=15)
        card.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
        
        value_label = ttk.Label(card, text=value, font=('Arial', 14, 'bold'), foreground=color)
        value_label.pack()

    def handle_prediction_error(self, error_msg):
        for widget in self.prediction_results.winfo_children():
            widget.destroy()

        error_label = ttk.Label(self.prediction_results, text=f"Prediction Failed\n\n{error_msg}", 
                               font=('Arial', 12), foreground='red', justify='center')
        error_label.pack(expand=True)

        self.log(f"Prediction Error: {error_msg}")
        self.finish_prediction()

    def handle_prediction_cancelled(self):
        for widget in self.prediction_results.winfo_children():
            widget.destroy()

        cancelled_label = ttk.Label(self.prediction_results, text="Prediction Cancelled", 
                                   font=('Arial', 14), foreground='#F57C00', justify='center')
        cancelled_label.pack(expand=True)

        self.log("Prediction cancelled successfully")
        self.finish_prediction()

    def finish_prediction(self):
        self.pred_running = False
        self.predict_btn.config(state='normal', text='MAKE PREDICTION')
        self.cancel_predict_btn.config(state='disabled')

    def update_quick_stats(self, result):
        ticker = result.get('ticker', 'N/A')
        
        stats_text = f"""
RESULTS SUMMARY FOR {ticker}
{'='*40}
Model Return: {result.get('model_return_pct', 0):.2f}%
Buy & Hold: {result.get('bh_return_pct', 0):.2f}%
Excess Return: {result.get('excess_return_pct', 0):.2f}%
Sharpe Ratio: {result.get('model_sharpe_ratio', 0):.3f}
Total Trades: {result.get('total_trades', 0)}
Completed: {datetime.now().strftime('%H:%M:%S')}
"""

        self.quick_stats.config(state='normal')
        self.quick_stats.delete(1.0, tk.END)
        self.quick_stats.insert(1.0, stats_text.strip())
        self.quick_stats.config(state='disabled')

    def update_analysis_tab(self, result):
        ticker = result.get('ticker', 'N/A')
        
        analysis_content = f"""
COMPLETE ANALYSIS FOR {ticker}
═══════════════════════════════════════════════════════════════════

PERFORMANCE METRICS
────────────────────
• Total Return: {result.get('model_return_pct', 0):.2f}%
• Annualized Return: {result.get('model_annualized_return', 0):.2f}%
• Volatility: {result.get('model_volatility_pct', 0):.2f}%
• Sharpe Ratio: {result.get('model_sharpe_ratio', 0):.3f}
• Maximum Drawdown: {result.get('model_max_drawdown_pct', 0):.2f}%
• Win Rate: {result.get('win_rate_pct', 0):.1f}%

BENCHMARK COMPARISON
─────────────────────
• Buy & Hold Return: {result.get('bh_return_pct', 0):.2f}%
• Excess Return: {result.get('excess_return_pct', 0):.2f}%
• Alpha Generation: {'Positive' if result.get('excess_return_pct', 0) > 0 else 'Negative'}

TRADING STATISTICS
───────────────────
• Total Trades: {result.get('total_trades', 0)}
• Profitable Trades: {result.get('profitable_trades', 0)}
• Total Signals: {result.get('strong_signals', 0)}
• Model Retrainings: {result.get('model_fits', 0)}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        self.analysis_text.config(state='normal')
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(1.0, analysis_content)
        self.analysis_text.config(state='disabled')

    def finish_run(self):
        self.running = False
        self.run_btn.config(state='normal', text='RUN BACKTEST')
        self.stop_btn.config(state='disabled')
        self.progress['value'] = 0
        self.progress_label.config(text="Ready to run backtest")

    def cleanup_and_exit(self):
        if self.running:
            self.stop_flag.set()
        if self.pred_running:
            self.pred_stop.set()
        
        self.destroyed = True
        gc.collect()
        self.quit()
        self.destroy()

if __name__ == "__main__":
    print("Starting MisuModels...")
    start_menu = StartMenu()
    start_menu.mainloop()
    print("Shutdown")