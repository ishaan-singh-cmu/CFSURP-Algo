import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, Callable
import logging
import re
from datetime import datetime
import threading
import gc

try:
    import src.core_model as aurora
except ImportError as e:
    print(f"Error importing aurora model: {e}")
    sys.exit(1)

class ProgressTracker:
    def __init__(self, progress_callback: Callable = None, stop_event: threading.Event = None):
        self.progress_callback = progress_callback
        self.stop_event = stop_event or threading.Event()
        self.current_progress = 0
        self.total_steps = 100
        self.current_step = 0
        self.phase = "Initializing"
        self.log_handler = None
        
        try:
            self.log_handler = ProgressLogHandler(self)
            logger = logging.getLogger()
            logger.addHandler(self.log_handler)
            logger.setLevel(logging.INFO)
        except Exception as e:
            print(f"Warning: Could not set up progress logging: {e}")

    def should_stop(self):
        return self.stop_event.is_set()

    def update_progress(self, current: int, total: int, phase: str = None):
        if self.should_stop():
            return
        try:
            self.current_step = current
            self.total_steps = max(total, 1)
            self.current_progress = min((current / total) * 100, 100)
            if phase:
                self.phase = phase
            if self.progress_callback:
                self.progress_callback(self.current_progress, self.phase, current, total)
        except Exception as e:
            print(f"Progress update error: {e}")

    def set_phase(self, phase: str):
        if self.should_stop():
            return
        try:
            self.phase = phase
            if self.progress_callback:
                self.progress_callback(self.current_progress, self.phase, self.current_step, self.total_steps)
        except Exception as e:
            print(f"Phase update error: {e}")

    def cleanup(self):
        try:
            if self.log_handler:
                logger = logging.getLogger()
                logger.removeHandler(self.log_handler)
                self.log_handler = None
        except Exception as e:
            print(f"Cleanup error: {e}")

class ProgressLogHandler(logging.Handler):
    def __init__(self, tracker: ProgressTracker):
        super().__init__()
        self.tracker = tracker

    def emit(self, record):
        try:
            if self.tracker.should_stop():
                return
            message = record.getMessage()
            
            progress_patterns = [
                (r'\w+:\s+(\d+)/(\d+)\s+\((\d+\.?\d*)%\)', "Backtesting"),
                (r'Training #(\d+)', "Training Models"),
                (r'Robust training #(\d+)', "Training Models"),
            ]
            
            for pattern, phase in progress_patterns:
                match = re.search(pattern, message)
                if match:
                    if r"(\d+)/(\d+)" in pattern:
                        current = int(match.group(1))
                        total = int(match.group(2))
                        self.tracker.update_progress(current, total, phase)
                    else:
                        self.tracker.set_phase(phase)
                    return
            
            phase_keywords = {
                "Preparing data": "Data Preparation",
                "Feature engineering": "Feature Engineering",
                "Model fitting": "Training Models",
                "SIGNAL:": "Signal Generation",
                "Position:": "Trade Execution",
                "RESULTS": "Results Complete"
            }
            
            for keyword, phase in phase_keywords.items():
                if keyword in message:
                    self.tracker.set_phase(phase)
                    break
        except Exception:
            pass

def run_regression_model(
    ticker: str,
    lookback_months: int = None,
    holdout_months: int = None,
    horizon_days: int = None,
    initial_capital: float = None,
    transaction_cost_rate: float = None,
    stop_loss_pct: float = None,
    take_profit_pct: float = None,
    max_position_size: float = None,
    pred_return_threshold: float = None,
    training_frequency: str = None,
    use_ensemble: bool = None,
    use_feature_selection: bool = None,
    train_start_date: str = None,
    trade_start_date: str = None,
    progress_callback: Callable = None,
    stop_event: threading.Event = None,
    **kwargs
) -> Dict:
    
    tracker = ProgressTracker(progress_callback, stop_event)
    
    try:
        tracker.set_phase("Initializing system")
        print(f"Starting backtest for {ticker}")
        
        if tracker.should_stop():
            return {'ticker': ticker, 'error': 'Stopped by user', 'stopped': True}
        
        config = aurora.TradingConfig()
        
        parameter_updates = {
            'lookback_months': lookback_months,
            'holdout_months': holdout_months,
            'horizon_days': horizon_days,
            'initial_capital': initial_capital,
            'training_frequency': training_frequency,
            'use_ensemble': use_ensemble,
            'use_feature_selection': use_feature_selection,
            'train_start_date': train_start_date,
            'trade_start_date': trade_start_date
        }
        
        percentage_params = {
            'transaction_cost_rate': transaction_cost_rate,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'max_position_size': max_position_size,
            'pred_return_threshold': pred_return_threshold
        }
        
        for param, value in parameter_updates.items():
            if value is not None:
                setattr(config, param, value)
        
        for param, value in percentage_params.items():
            if value is not None:
                setattr(config, param, value / 100.0)
        
        config.tickers = [ticker]
        tracker.set_phase("Configuration complete")
        tracker.update_progress(10, 100, "Starting backtest")
        
        if tracker.should_stop():
            return {'ticker': ticker, 'error': 'Stopped by user', 'stopped': True}
        
        print(f"Configuration:")
        print(f" - Ticker: {ticker}")
        print(f" - Capital: ${config.initial_capital:,.0f}")
        print(f" - Max Position: {config.max_position_size:.1%}")
        print(f" - Stop Loss: {config.stop_loss_pct:.1%}")
        print(f" - Take Profit: {config.take_profit_pct:.1%}")
        
        if tracker.should_stop():
            return {'ticker': ticker, 'error': 'Stopped by user', 'stopped': True}
        
        backtester = aurora.Backtester(config)
        tracker.update_progress(15, 100, "Running backtest")
        
        if tracker.should_stop():
            return {'ticker': ticker, 'error': 'Stopped by user', 'stopped': True}
        
        result = backtester.run_backtest(ticker)
        
        if tracker.should_stop():
            return {'ticker': ticker, 'error': 'Stopped by user', 'stopped': True}
        
        tracker.update_progress(100, 100, "Backtest complete")
        
        if 'error' in result:
            print(f"Backtest failed: {result['error']}")
            return {
                'ticker': ticker,
                'error': result['error'],
                'model_final_equity': None,
                'model_return_pct': None,
                'model_annualized_return': None,
                'model_max_drawdown_pct': None,
                'model_sharpe_ratio': None,
                'bh_return_pct': None,
                'bh_annualized_return': None,
                'bh_max_drawdown_pct': None,
                'excess_return_pct': None,
                'win_rate_pct': None,
                'total_trades': 0,
                'total_predictions': 0,
                'strong_signals': 0,
                'model_fits': 0,
                'training_frequency': config.training_frequency,
                'chart_files': [],
                'equity_curve': pd.Series(),
                'trades_data': pd.DataFrame(),
                'model_stats': {},
                'trained_model': None,
                'trained_backtester': None,
                'model_config': None
            }
        
        metrics = result.get('metrics', {})
        model_stats = result.get('model_stats', {})
        final_capital = result.get('final_capital', config.initial_capital)
        trades_df = result.get('trades', pd.DataFrame())
        equity_curve = result.get('equity_curve', pd.Series())
        chart_files = result.get('chart_files', [])
        
        trained_model = None
        trained_backtester = None
        if hasattr(backtester, 'model') and backtester.model.is_fitted:
            trained_model = backtester.model
            trained_backtester = backtester
            print("Model cached for predictions")
        else:
            print("No trained model available")
        
        if 'buy_hold_return_pct' not in metrics and not tracker.should_stop():
            tracker.set_phase("Calculating buy & hold")
            try:
                import yfinance as yf
                end_date = datetime.now()
                start_date = pd.Timestamp(config.trade_start_date)
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if not data.empty:
                    bh_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                    metrics['buy_hold_return_pct'] = bh_return
                    
                    bh_returns = data['Close'].pct_change().dropna()
                    years = len(data) / 252
                    if years > 0:
                        bh_ann_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) ** (1/years) - 1) * 100
                        metrics['bh_annualized_return'] = bh_ann_return
                    
                    bh_peak = data['Close'].cummax()
                    bh_drawdown = (bh_peak - data['Close']) / bh_peak
                    metrics['bh_max_drawdown_pct'] = bh_drawdown.max() * 100
                    
            except Exception as e:
                print(f"Warning: Could not calculate buy and hold metrics: {e}")
                metrics.update({
                    'buy_hold_return_pct': 0,
                    'bh_annualized_return': 0,
                    'bh_max_drawdown_pct': 0
                })
        
        if not equity_curve.empty:
            returns = equity_curve.pct_change().dropna()
            if len(returns) > 0:
                if metrics.get('max_drawdown_pct', 0) > 0:
                    metrics['calmar_ratio'] = metrics.get('annualized_return_pct', 0) / metrics['max_drawdown_pct']
                else:
                    metrics['calmar_ratio'] = 0
                
                negative_returns = returns[returns < 0]
                if len(negative_returns) > 0:
                    downside_deviation = negative_returns.std() * np.sqrt(252)
                    if downside_deviation > 0:
                        excess_return = metrics.get('annualized_return_pct', 0) / 100 - 0.02
                        metrics['sortino_ratio'] = excess_return / downside_deviation
                    else:
                        metrics['sortino_ratio'] = 0
        
        if not trades_df.empty and 'pnl' in trades_df.columns:
            profitable_trades = trades_df[trades_df['pnl'] > 0] if 'pnl' in trades_df.columns else pd.DataFrame()
            losing_trades = trades_df[trades_df['pnl'] < 0] if 'pnl' in trades_df.columns else pd.DataFrame()
            
            metrics['total_trades'] = len(trades_df)
            metrics['profitable_trades'] = len(profitable_trades)
            metrics['losing_trades'] = len(losing_trades)
            
            if len(profitable_trades) > 0:
                metrics['avg_win'] = profitable_trades['pnl'].mean()
                metrics['largest_win'] = profitable_trades['pnl'].max()
            
            if len(losing_trades) > 0:
                metrics['avg_loss'] = losing_trades['pnl'].mean()
                metrics['largest_loss'] = losing_trades['pnl'].min()
            
            total_profit = profitable_trades['pnl'].sum() if len(profitable_trades) > 0 else 0
            total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1
            metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf')
        
        formatted_result = {
            'ticker': ticker,
            'model_final_equity': final_capital,
            'model_return_pct': metrics.get('total_return_pct', 0),
            'model_annualized_return': metrics.get('annualized_return_pct', 0),
            'model_max_drawdown_pct': metrics.get('max_drawdown_pct', 0),
            'model_sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'model_volatility_pct': metrics.get('annualized_volatility_pct', 0),
            'calmar_ratio': metrics.get('calmar_ratio', 0),
            'sortino_ratio': metrics.get('sortino_ratio', 0),
            'bh_return_pct': metrics.get('buy_hold_return_pct', 0),
            'bh_annualized_return': metrics.get('bh_annualized_return', 0),
            'bh_max_drawdown_pct': metrics.get('bh_max_drawdown_pct', 0),
            'excess_return_pct': metrics.get('excess_return_pct', 0),
            'win_rate_pct': metrics.get('win_rate_pct', 0),
            'total_trades': len(trades_df),
            'profitable_trades': metrics.get('profitable_trades', 0),
            'total_predictions': model_stats.get('total_predictions', 0),
            'strong_signals': model_stats.get('strong_signals', 0),
            'model_fits': model_stats.get('model_fits', 0),
            'training_frequency': model_stats.get('training_frequency', 'weekly'),
            'chart_files': chart_files,
            'equity_curve': equity_curve,
            'trades_data': trades_df,
            'model_stats': model_stats,
            'all_metrics': metrics,
            'trained_model': trained_model,
            'trained_backtester': trained_backtester,
            'model_config': config,
            'error': None
        }
        
        print(f"{ticker} backtest completed")
        print(f"Total Return: {formatted_result['model_return_pct']:.2f}%")
        print(f"Trades: {formatted_result['total_trades']}")
        print(f"Signals: {formatted_result['strong_signals']}")
        
        if trained_model:
            print("Model cached for predictions")
        
        return formatted_result
        
    except Exception as e:
        print(f"Error in backtest for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'ticker': ticker,
            'error': str(e),
            'model_final_equity': None,
            'model_return_pct': None,
            'model_annualized_return': None,
            'model_max_drawdown_pct': None,
            'model_sharpe_ratio': None,
            'bh_return_pct': None,
            'bh_annualized_return': None,
            'bh_max_drawdown_pct': None,
            'excess_return_pct': None,
            'win_rate_pct': None,
            'total_trades': 0,
            'total_predictions': 0,
            'strong_signals': 0,
            'model_fits': 0,
            'training_frequency': 'weekly',
            'chart_files': [],
            'equity_curve': pd.Series(),
            'trades_data': pd.DataFrame(),
            'model_stats': {},
            'trained_model': None,
            'trained_backtester': None,
            'model_config': None
        }
    
    finally:
        try:
            tracker.cleanup()
            gc.collect()
        except Exception as e:
            print(f"Cleanup error: {e}")

def test_model(ticker: str = "AAPL") -> None:
    def progress_callback(progress, phase, current, total):
        print(f"Progress: {progress:.1f}% - {phase} ({current}/{total})")
    
    print(f"Testing model interface with {ticker}")
    
    result = run_regression_model(
        ticker=ticker,
        lookback_months=6,
        holdout_months=3,
        horizon_days=5,
        initial_capital=100000,
        stop_loss_pct=3.0,
        take_profit_pct=6.0,
        max_position_size=5.0,
        progress_callback=progress_callback
    )
    
    print("Test Results:")
    for key, value in result.items():
        if value is not None and key not in ['equity_curve', 'trades_data', 'chart_files', 'trained_model', 'trained_backtester', 'model_config']:
            if isinstance(value, float):
                print(f" {key}: {value:.4f}")
            else:
                print(f" {key}: {value}")
    
    if result.get('trained_model'):
        print("Model available for predictions")
    else:
        print("No model available")

if __name__ == "__main__":
    test_model()