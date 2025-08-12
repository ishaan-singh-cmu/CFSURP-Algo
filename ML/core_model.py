import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from scipy import stats

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logging.warning("TA library not available, using simplified technical indicators")

@dataclass
class TradingConfig:
    tickers: List[str] = None
    initial_capital: float = 1_000_000.0
    train_start_date: str = "2015-01-01"
    trade_start_date: str = "2017-01-01"
    
    horizon_days: int = 5
    lookback_months: int = 12
    holdout_months: int = 3
    pred_return_threshold: float = 0.005
    
    training_frequency: str = "weekly"
    
    max_position_size: float = 0.05
    max_portfolio_leverage: float = 0.95
    max_short_exposure: float = 0.15
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06
    volatility_lookback: int = 20
    
    transaction_cost_rate: float = 0.001
    short_borrow_rate: float = 0.02
    
    use_ensemble: bool = True
    use_feature_selection: bool = True
    cv_folds: int = 3
    
    output_root: str = "newml"
    
    def __post_init__(self):
        if self.tickers is None:
            self.tickers = ['AAPL', 'TSLA', 'META', 'NVDA']
        os.makedirs(self.output_root, exist_ok=True)

class FeatureEngineer:
    def __init__(self):
        self.feature_names = []

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build feature set from price data"""
        df = df.copy()
        
        if len(df) < 100:
            raise ValueError("Need more data")
        
        logging.info(f"Creating features with {len(df)} rows")
        
        df = self._clean_data(df)
        df = self._price_features(df)
        df = self._technical_features(df)
        df = self._vol_features(df)
        df = self._volume_features(df)
        df = self._market_regime(df)
        df = self._time_features(df)
        df = self._final_cleanup(df)
        
        # target variable
        df['future_return'] = df['Close'].shift(-5) / df['Close'] - 1
        
        logging.info(f"Created {len([c for c in df.columns if c not in ['Open','High','Low','Close','Volume','future_return']])} features")
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.ffill().bfill()
        df = df[df['Volume'] > 0]
        df['returns'] = df['Close'].pct_change()
        df['returns'] = df['returns'].replace([np.inf, -np.inf], np.nan).fillna(0)
        df['range_pct'] = (df['High'] - df['Low']) / df['Close'].replace(0, np.nan)
        df['body_pct'] = abs(df['Close'] - df['Open']) / df['Close'].replace(0, np.nan)
        return df

    def _price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        periods = [1, 3, 5, 10, 20, 50]
        
        for p in periods:
            if len(df) > p:
                df[f'momentum_{p}'] = df['Close'].pct_change(p)
                df[f'momentum_{p}'] = df[f'momentum_{p}'].replace([np.inf, -np.inf], np.nan)
                df[f'momentum_{p}_rank'] = df[f'momentum_{p}'].rolling(min(100, len(df)//4)).rank(pct=True)
        
        # exponential moving averages
        for p in [5, 10, 20, 50]:
            if len(df) > p:
                df[f'ema_{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
                df[f'price_ema_{p}_ratio'] = df['Close'] / df[f'ema_{p}'].replace(0, np.nan)
        
        return df

    def _technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # RSI calculation
        for period in [14, 30]:
            if len(df) > period:
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                rs = gain / loss.replace(0, np.nan)
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        if len(df) > 26:
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        if len(df) > 20:
            sma = df['Close'].rolling(20).mean()
            std = df['Close'].rolling(20).std()
            df['bb_upper'] = sma + (std * 2)
            df['bb_lower'] = sma - (std * 2)
            df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
        
        return df

    def _vol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        windows = [5, 10, 20, 50]
        for w in windows:
            if len(df) > w:
                df[f'volatility_{w}'] = df['returns'].rolling(w).std()
                df[f'volatility_{w}_rank'] = df[f'volatility_{w}'].rolling(min(60, len(df)//4)).rank(pct=True)
        return df

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for window in [5, 10, 20]:
            if len(df) > window:
                df[f'volume_sma_{window}'] = df['Volume'].rolling(window).mean()
                df[f'volume_ratio_{window}'] = df['Volume'] / df[f'volume_sma_{window}'].replace(0, np.nan)
        return df

    def _market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        # moving averages for trend detection
        for ma in [10, 20, 50]:
            if len(df) > ma:
                df[f'ma_{ma}'] = df['Close'].rolling(ma).mean()
                df[f'price_ma_{ma}_ratio'] = df['Close'] / df[f'ma_{ma}'].replace(0, np.nan)
        
        if len(df) > 50:
            df['trend_up'] = (df['ma_10'] > df['ma_20']).astype(int)
            df['trend_strong'] = (df['ma_20'] > df['ma_50']).astype(int)
        
        return df

    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        return df

    def _final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill().fillna(0)
        
        # remove constant columns
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            df = df.drop(columns=constant_cols)
            logging.info(f"Removed {len(constant_cols)} constant features")
        
        return df

class EnsembleModel:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.models = {}
        self.weights = {}
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.selected_features = None
        self.fit_count = 0
        self.is_fitted = False

    def _build_models(self):
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'gbm': GradientBoostingRegressor(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            ),
            'ridge': Ridge(alpha=5.0, random_state=42)
        }

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        self.fit_count += 1
        
        logging.info(f"Model fitting #{self.fit_count}")
        logging.info(f"Data shape: {X.shape[0]} samples, {X.shape[1]} features")
        logging.info(f"Target stats: mean={y.mean():.4f}, std={y.std():.4f}")
        
        # clean invalid data
        if np.isnan(X).any() or np.isnan(y).any():
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]
            logging.info(f"Cleaned to {X.shape[0]} valid samples")
        
        if len(X) < 30:
            raise ValueError(f"Not enough samples: {len(X)}")
        
        self._build_models()
        
        # feature selection
        if self.config.use_feature_selection and feature_names and X.shape[1] > 15:
            selector = SelectKBest(score_func=f_regression, k=min(15, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            self.feature_selector = selector.get_support(indices=True)
            self.selected_features = [feature_names[i] for i in self.feature_selector]
            logging.info(f"Selected {len(self.selected_features)} features")
        else:
            X_selected = X
            self.selected_features = feature_names
            self.feature_selector = None
        
        # scale features
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # train and evaluate models
        cv_scores = {}
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        logging.info(f"Training {len(self.models)} models")
        
        for name, model in self.models.items():
            logging.info(f"Training {name.upper()}...")
            try:
                scores = cross_val_score(
                    model, X_scaled, y,
                    cv=tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=1
                )
                cv_scores[name] = -scores.mean()
                model.fit(X_scaled, y)
                logging.info(f"{name.upper()}: CV MSE = {cv_scores[name]:.6f}")
            except Exception as e:
                logging.warning(f"{name.upper()} failed: {e}")
                cv_scores[name] = float('inf')
        
        # calculate model weights
        valid_scores = {k: v for k, v in cv_scores.items() if v != float('inf')}
        if valid_scores:
            inv_scores = {k: 1.0/(v + 1e-8) for k, v in valid_scores.items()}
            total_inv = sum(inv_scores.values())
            self.weights = {k: v/total_inv for k, v in inv_scores.items()}
            logging.info("Model weights:")
            for name, weight in sorted(self.weights.items(), key=lambda x: x[1], reverse=True):
                logging.info(f"{name.upper()}: {weight:.3f}")
        else:
            valid_models = [k for k, v in cv_scores.items() if v != float('inf')]
            if valid_models:
                self.weights = {k: 1.0/len(valid_models) for k in valid_models}
            else:
                raise ValueError("All models failed")
        
        self.is_fitted = True
        logging.info("Model training complete")

    def _generate_recommendation(self, predicted_return: float, confidence: float) -> str:
        if confidence < 30:
            return "LOW CONFIDENCE - Hold"
        elif predicted_return > 0.02:
            return "STRONG BUY" if confidence > 70 else "BUY"
        elif predicted_return > 0.005:
            return "BUY" if confidence > 60 else "WEAK BUY"
        elif predicted_return > -0.005:
            return "HOLD"
        elif predicted_return > -0.02:
            return "SELL" if confidence > 60 else "WEAK SELL"
        else:
            return "STRONG SELL" if confidence > 70 else "SELL"

    def predict_future(self, current_data: pd.DataFrame, days_ahead: int = 5) -> Dict:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        feature_cols = [col for col in current_data.columns
                       if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'future_return']]
        
        if len(feature_cols) == 0:
            raise ValueError("No features available")
        
        latest_features = current_data[feature_cols].iloc[-1:].values
        
        # handle feature selection
        if self.feature_selector is not None:
            max_feature_idx = latest_features.shape[1] - 1
            valid_indices = [idx for idx in self.feature_selector if idx <= max_feature_idx]
            
            if len(valid_indices) == 0:
                logging.warning("Feature selector indices out of bounds, using all features")
                latest_features_selected = latest_features
            else:
                latest_features_selected = latest_features[:, valid_indices]
                if len(valid_indices) < len(self.feature_selector):
                    logging.warning(f"Using {len(valid_indices)}/{len(self.feature_selector)} selected features")
        else:
            latest_features_selected = latest_features
        
        latest_features_scaled = self.scaler.transform(latest_features_selected)
        predicted_return = self._predict_with_fallback(latest_features_scaled)
        
        current_price = current_data['Close'].iloc[-1]
        predicted_price = current_price * (1 + predicted_return)
        
        # confidence calculation (simplified)
        recent_volatility = current_data['Close'].pct_change().tail(20).std()
        model_consistency = len([w for w in self.weights.values() if w > 0.1])
        base_confidence = min(70, 30 + model_consistency * 15)
        vol_penalty = min(20, recent_volatility * 1000)
        confidence = max(10, base_confidence - vol_penalty)
        
        recommendation = self._generate_recommendation(predicted_return, confidence)
        
        return {
            'predicted_return_pct': predicted_return * 100,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'confidence_pct': confidence,
            'recommendation': recommendation,
            'signal_strength': abs(predicted_return) / self.config.pred_return_threshold,
            'days_ahead': days_ahead
        }

    def _predict_with_fallback(self, X: np.ndarray) -> float:
        try:
            if self.feature_selector is not None:
                expected_features = len(self.feature_selector)
                actual_features = X.shape[1]
                if actual_features != expected_features:
                    logging.warning(f"Feature count mismatch: expected {expected_features}, got {actual_features}")
            
            predictions = []
            total_weight = 0
            
            for name, model in self.models.items():
                if name in self.weights and self.weights[name] > 0:
                    try:
                        pred = model.predict(X)[0]
                        predictions.append(pred * self.weights[name])
                        total_weight += self.weights[name]
                    except Exception as e:
                        logging.warning(f"Model {name} prediction failed: {e}")
                        continue
            
            if predictions and total_weight > 0:
                ensemble_pred = sum(predictions) / total_weight
                return np.clip(ensemble_pred, -0.1, 0.1)
            else:
                logging.warning("All model predictions failed")
                return 0.0
        
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        if self.feature_selector is not None:
            X_selected = X[:, self.feature_selector]
        else:
            X_selected = X
        
        X_scaled = self.scaler.transform(X_selected)
        
        predictions = []
        total_weight = 0
        
        for name, model in self.models.items():
            if name in self.weights and self.weights[name] > 0:
                try:
                    pred = model.predict(X_scaled)
                    predictions.append(pred * self.weights[name])
                    total_weight += self.weights[name]
                except Exception as e:
                    logging.warning(f"Prediction failed for {name}: {e}")
        
        if predictions and total_weight > 0:
            ensemble_pred = np.sum(predictions, axis=0) / total_weight
            ensemble_pred = np.clip(ensemble_pred, -0.1, 0.1)
            if len(ensemble_pred) == 1:
                logging.info(f"Prediction: {ensemble_pred[0]:.4f}")
            return ensemble_pred
        else:
            logging.warning("All predictions failed")
            return np.zeros(X.shape[0])

class RiskManager:
    def __init__(self, config: TradingConfig):
        self.config = config

    def calculate_position_size(self, predicted_return: float, volatility: float,
                              current_capital: float, current_price: float,
                              signal_strength: float = 1.0) -> Tuple[float, str]:
        
        max_dollar_allocation = current_capital * self.config.max_position_size
        
        strength_adj = min(max(signal_strength, 0.1), 2.0)
        vol_adj = 1.0 / (1.0 + volatility * 20) if volatility > 0 else 1.0
        
        dollar_allocation = max_dollar_allocation * strength_adj * vol_adj
        dollar_allocation = max(dollar_allocation, current_capital * 0.005)
        dollar_allocation = min(dollar_allocation, current_capital * 0.1)
        
        direction = "LONG" if predicted_return > 0 else "SHORT"
        shares = dollar_allocation / current_price if current_price > 0 else 0
        
        return shares, direction

    def calculate_stop_loss_take_profit(self, entry_price: float, direction: str,
                                      volatility: float, predicted_return: float,
                                      signal_strength: float = 1.0) -> Tuple[float, float]:
        vol_multiplier = max(0.5, min(2.0, volatility * 30)) if volatility > 0 else 1.0
        
        if direction == "LONG":
            stop_loss = entry_price * (1 - self.config.stop_loss_pct * vol_multiplier)
            take_profit = entry_price * (1 + self.config.take_profit_pct)
        else:
            stop_loss = entry_price * (1 + self.config.stop_loss_pct * vol_multiplier)
            take_profit = entry_price * (1 - self.config.take_profit_pct)
        
        return stop_loss, take_profit

class TrainingScheduler:
    def __init__(self, frequency: str):
        self.frequency = frequency.lower()
        self.last_training_date = None

    def should_train(self, current_date: pd.Timestamp) -> bool:
        if self.last_training_date is None:
            self.last_training_date = current_date
            return True
        
        days_since = (current_date - self.last_training_date).days
        
        frequency_map = {
            "daily": 1,
            "weekly": 7,
            "biweekly": 14,
            "monthly": 30,
            "bimonthly": 60
        }
        
        threshold = frequency_map.get(self.frequency, 7)
        should_retrain = days_since >= threshold
        
        if should_retrain:
            self.last_training_date = current_date
        
        return should_retrain

def create_charts(equity_curve: pd.Series, price_data: pd.DataFrame,
                 trades_data: pd.DataFrame, ticker: str, config: TradingConfig) -> List[str]:
    chart_files = []
    
    try:
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        
        trade_start = pd.Timestamp(config.trade_start_date)
        price_subset = price_data[price_data.index >= trade_start]
        
        if price_subset.empty or equity_curve.empty:
            logging.warning(f"No data to plot for {ticker}")
            return []
        
        common_dates = equity_curve.index.intersection(price_subset.index)
        if len(common_dates) == 0:
            logging.warning(f"No common dates for {ticker}")
            return []
        
        equity_aligned = equity_curve.loc[common_dates]
        price_aligned = price_subset.loc[common_dates]
        
        # buy and hold baseline
        initial_price = price_aligned['Close'].iloc[0]
        shares_bh = config.initial_capital / initial_price
        buy_hold_equity = shares_bh * price_aligned['Close']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # performance comparison
        ax1.plot(equity_aligned.index, equity_aligned.values,
                label='ML Strategy', linewidth=2, color='blue', alpha=0.9)
        ax1.plot(buy_hold_equity.index, buy_hold_equity.values,
                label='Buy & Hold', linewidth=2, color='red', alpha=0.7)
        ax1.set_title(f'{ticker} - ML Strategy vs Buy & Hold', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # performance stats
        final_ml = equity_aligned.iloc[-1]
        final_bh = buy_hold_equity.iloc[-1]
        ml_return = (final_ml / config.initial_capital - 1) * 100
        bh_return = (final_bh / config.initial_capital - 1) * 100
        
        stats_text = f'ML Strategy: {ml_return:.1f}%\nBuy & Hold: {bh_return:.1f}%\nOutperformance: {ml_return - bh_return:.1f}%'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                verticalalignment='top', fontsize=11, fontweight='bold')
        
        # drawdown analysis
        ml_peak = equity_aligned.cummax()
        ml_drawdown = (ml_peak - equity_aligned) / ml_peak * 100
        bh_peak = buy_hold_equity.cummax()
        bh_drawdown = (bh_peak - buy_hold_equity) / bh_peak * 100
        
        ax2.fill_between(ml_drawdown.index, ml_drawdown.values, 0,
                        alpha=0.3, color='blue', label='ML Strategy DD')
        ax2.fill_between(bh_drawdown.index, bh_drawdown.values, 0,
                        alpha=0.3, color='red', label='Buy & Hold DD')
        ax2.set_title(f'{ticker} - Drawdown Analysis', fontsize=14)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # trade analysis
        if not trades_data.empty and 'pnl' in trades_data.columns:
            ax3.hist(trades_data['pnl'], bins=20, alpha=0.7, color='green', edgecolor='black')
            ax3.axvline(trades_data['pnl'].mean(), color='red', linestyle='--',
                       linewidth=2, label=f'Mean P&L: ${trades_data["pnl"].mean():.0f}')
            ax3.set_title(f'{ticker} - Trade P&L Distribution', fontsize=14)
            ax3.set_xlabel('P&L ($)')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Trade Data Available', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title(f'{ticker} - Trade Analysis', fontsize=14)
        
        # returns comparison
        ml_returns = equity_aligned.pct_change().dropna()
        bh_returns = buy_hold_equity.pct_change().dropna()
        
        ax4.hist(ml_returns * 100, bins=30, alpha=0.6, label='ML Strategy', color='blue')
        ax4.hist(bh_returns * 100, bins=30, alpha=0.6, label='Buy & Hold', color='red')
        ax4.set_title(f'{ticker} - Daily Returns Distribution', fontsize=14)
        ax4.set_xlabel('Daily Return (%)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = os.path.join(config.output_root, f'{ticker}_performance.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        chart_files.append(filename)
        logging.info(f"Chart saved: {filename}")
        
        return chart_files
    
    except Exception as e:
        logging.error(f"Chart creation failed for {ticker}: {e}")
        return []

class Backtester:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer()
        self.risk_manager = RiskManager(config)
        self.model = EnsembleModel(config)
        self.training_scheduler = TrainingScheduler(config.training_frequency)

    def prepare_data(self, ticker: str) -> pd.DataFrame:
        logging.info(f"Preparing data for {ticker}...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10*365)
        
        # fetch data with retry logic
        for attempt in range(3):
            try:
                df = yf.download(ticker, start=start_date, end=end_date,
                               interval='1d', auto_adjust=True, progress=False)
                if not df.empty and len(df) > 1000:
                    break
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                if attempt == 2:
                    raise
        
        if df is None or df.empty or len(df) < 1000:
            raise ValueError(f"Insufficient data for {ticker}")
        
        # handle multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # data validation
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns for {ticker}: {missing_cols}")
        
        # clean data
        df = df.dropna(subset=required_cols)
        df = df[df['Volume'] > 0]
        df = df[(df['High'] >= df['Low']) & (df['High'] >= df['Close']) & (df['Low'] <= df['Close'])]
        
        # remove outliers
        for col in required_cols[:-1]:
            Q1 = df[col].quantile(0.01)
            Q99 = df[col].quantile(0.99)
            df = df[(df[col] >= Q1) & (df[col] <= Q99)]
        
        if len(df) < 500:
            raise ValueError(f"Insufficient clean data for {ticker}: {len(df)} rows")
        
        logging.info(f"Basic data prepared: {len(df)} rows")
        
        # create features
        df = self.feature_engineer.create_features(df)
        df = df.dropna()
        
        if len(df) < 300:
            raise ValueError(f"Insufficient data after feature engineering for {ticker}: {len(df)} rows")
        
        logging.info(f"Data ready: {len(df)} final rows")
        return df

    def run_backtest(self, ticker: str) -> Dict:
        try:
            df = self.prepare_data(ticker)
            
            train_start = pd.Timestamp(self.config.train_start_date)
            trade_start = pd.Timestamp(self.config.trade_start_date)
            
            df = df[df.index >= train_start].copy()
            df = df.sort_index()
            
            if len(df[df.index >= trade_start]) < 100:
                raise ValueError(f"Insufficient trading data for {ticker}")
            
            # initialize tracking variables
            capital = self.config.initial_capital
            cash = capital
            equity_series = {}
            trades = []
            current_positions = {}
            prediction_count = 0
            signal_count = 0
            training_count = 0
            
            # feature columns
            price_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'future_return']
            feature_cols = [col for col in df.columns if col not in price_cols]
            
            logging.info(f"Starting backtest for {ticker}")
            logging.info(f"Features: {len(feature_cols)}")
            logging.info(f"Period: {trade_start.date()} to {df.index.max().date()}")
            logging.info(f"Initial capital: ${capital:,.0f}")
            
            total_days = len(df[df.index >= trade_start])
            current_day = 0
            
            # main loop
            for i, (date, row) in enumerate(df.iterrows()):
                if date < trade_start:
                    equity_series[date] = capital
                    continue
                
                current_day += 1
                if current_day % 100 == 0:
                    progress = (current_day / total_days) * 100
                    logging.info(f"{ticker}: {current_day}/{total_days} ({progress:.1f}%)")
                
                # check exits
                cash = self._check_exits(current_positions, row, cash, date, trades)
                
                # calculate current equity
                position_value = sum(pos['shares'] * row['Close'] for pos in current_positions.values())
                current_equity = cash + position_value
                equity_series[date] = current_equity
                
                # get training data
                lookback_start = date - pd.DateOffset(months=self.config.lookback_months)
                train_data = df.loc[(df.index >= lookback_start) & (df.index < date)]
                
                if len(train_data) < 100:
                    continue
                
                # training
                should_retrain = self.training_scheduler.should_train(date)
                if should_retrain:
                    X_train = train_data[feature_cols].values
                    y_train = train_data['future_return'].values
                    
                    # clean training data
                    valid_mask = ~(np.isnan(y_train) | np.isinf(y_train))
                    for col_idx in range(X_train.shape[1]):
                        valid_mask &= ~(np.isnan(X_train[:, col_idx]) | np.isinf(X_train[:, col_idx]))
                    
                    X_train = X_train[valid_mask]
                    y_train = y_train[valid_mask]
                    
                    if len(X_train) < 50:
                        continue
                    
                    try:
                        training_count += 1
                        logging.info(f"{date.date()} - Training #{training_count}")
                        self.model.fit(X_train, y_train, feature_cols)
                    except Exception as e:
                        logging.warning(f"Training failed: {e}")
                        continue
                
                # prediction
                if training_count == 0 or len(current_positions) >= 5:
                    continue
                
                X_current = row[feature_cols].values.reshape(1, -1)
                if np.isnan(X_current).any() or np.isinf(X_current).any():
                    continue
                
                try:
                    predicted_return = self.model.predict(X_current)[0]
                    prediction_count += 1
                except Exception as e:
                    continue
                
                # signal evaluation
                signal_strength = abs(predicted_return) / self.config.pred_return_threshold
                if signal_strength < 1.0:
                    continue
                
                signal_count += 1
                
                # position sizing
                volatility = train_data['returns'].std()
                shares, direction = self.risk_manager.calculate_position_size(
                    predicted_return, volatility, current_equity, row['Close'], signal_strength
                )
                
                if shares <= 0:
                    continue
                
                # calculate costs
                position_value = shares * row['Close']
                transaction_cost = position_value * self.config.transaction_cost_rate
                
                # check cash availability
                if position_value + transaction_cost > cash:
                    continue
                
                # enter position
                position_id = f"{ticker}_{date.strftime('%Y%m%d')}_{len(trades)}"
                current_positions[position_id] = {
                    'ticker': ticker,
                    'entry_date': date,
                    'entry_price': row['Close'],
                    'shares': shares,
                    'direction': direction,
                    'predicted_return': predicted_return,
                    'signal_strength': signal_strength,
                    'exit_date': date + pd.Timedelta(days=self.config.horizon_days),
                    'position_value': position_value
                }
                
                # update cash
                cash -= (position_value + transaction_cost)
                
                trades.append({
                    'position_id': position_id,
                    'entry_date': date,
                    'entry_price': row['Close'],
                    'shares': shares,
                    'direction': direction,
                    'predicted_return': predicted_return,
                    'signal_strength': signal_strength,
                    'position_value': position_value,
                    'transaction_cost': transaction_cost
                })
                
                logging.info(f"Position: {direction} {shares:.1f} @ ${row['Close']:.2f}")
            
            # close remaining positions
            final_date = df.index[-1]
            final_row = df.iloc[-1]
            cash = self._check_exits(current_positions, final_row, cash, final_date, trades, force_close=True)
            
            final_equity = cash
            
            # results
            equity_df = pd.Series(equity_series).sort_index()
            trades_df = pd.DataFrame(trades)
            metrics = self._calculate_metrics(equity_df, df, ticker)
            chart_files = create_charts(equity_df, df, trades_df, ticker, self.config)
            
            total_return = (final_equity / self.config.initial_capital - 1) * 100
            
            logging.info(f"Results for {ticker}:")
            logging.info(f"Predictions: {prediction_count}, Signals: {signal_count}")
            logging.info(f"Trades: {len(trades)}, Training sessions: {training_count}")
            logging.info(f"Total return: {total_return:.2f}%")
            
            return {
                'ticker': ticker,
                'metrics': metrics,
                'equity_curve': equity_df,
                'trades': trades_df,
                'final_capital': final_equity,
                'chart_files': chart_files,
                'model_stats': {
                    'total_predictions': prediction_count,
                    'strong_signals': signal_count,
                    'model_fits': training_count,
                    'training_frequency': self.config.training_frequency
                }
            }
        
        except Exception as e:
            logging.error(f"Backtest failed for {ticker}: {e}")
            return {
                'ticker': ticker,
                'error': str(e),
                'metrics': {},
                'equity_curve': pd.Series(),
                'trades': pd.DataFrame(),
                'final_capital': self.config.initial_capital,
                'chart_files': [],
                'model_stats': {
                    'total_predictions': 0,
                    'strong_signals': 0,
                    'model_fits': 0,
                    'training_frequency': self.config.training_frequency
                }
            }

    def _check_exits(self, positions: Dict, current_row, cash: float,
                    current_date, trades: List, force_close: bool = False) -> float:
        positions_to_remove = []
        
        for pos_id, position in positions.items():
            should_exit = False
            exit_reason = ""
            
            if force_close:
                should_exit = True
                exit_reason = "force_close"
            elif current_date >= position['exit_date']:
                should_exit = True
                exit_reason = "time_exit"
            
            if should_exit:
                # calculate proceeds
                if position['direction'] == 'LONG':
                    proceeds = position['shares'] * current_row['Close']
                else:
                    proceeds = position['shares'] * (2 * position['entry_price'] - current_row['Close'])
                
                exit_cost = proceeds * self.config.transaction_cost_rate
                net_proceeds = proceeds - exit_cost
                cash += net_proceeds
                
                pnl = net_proceeds - position['position_value']
                return_pct = pnl / position['position_value'] * 100
                
                # update trade record
                for trade in trades:
                    if trade['position_id'] == pos_id:
                        trade.update({
                            'exit_date': current_date,
                            'exit_price': current_row['Close'],
                            'exit_reason': exit_reason,
                            'pnl': pnl,
                            'return_pct': return_pct,
                            'exit_cost': exit_cost,
                            'proceeds': proceeds
                        })
                        break
                
                positions_to_remove.append(pos_id)
                
                if abs(pnl) > 1000:
                    logging.info(f"{exit_reason}: {position['direction']} ${pnl:,.0f} ({return_pct:.1f}%)")
        
        # remove closed positions
        for pos_id in positions_to_remove:
            del positions[pos_id]
        
        return cash

    def _calculate_metrics(self, equity_curve: pd.Series, price_data: pd.DataFrame, ticker: str) -> Dict:
        metrics = {}
        
        if equity_curve.empty:
            return metrics
        
        initial_capital = equity_curve.iloc[0]
        final_capital = equity_curve.iloc[-1]
        total_return = (final_capital / initial_capital - 1) * 100
        
        n_days = len(equity_curve)
        if n_days > 1:
            daily_returns = equity_curve.pct_change().dropna()
            years = n_days / 252
            annualized_return = (final_capital / initial_capital) ** (1/years) - 1
            annualized_volatility = daily_returns.std() * np.sqrt(252)
            
            risk_free_rate = 0.02
            excess_return = annualized_return - risk_free_rate
            sharpe_ratio = excess_return / annualized_volatility if annualized_volatility > 0 else 0
            
            peak = equity_curve.cummax()
            drawdown = (peak - equity_curve) / peak
            max_drawdown = drawdown.max() * 100
            
            positive_returns = daily_returns[daily_returns > 0]
            win_rate = len(positive_returns) / len(daily_returns) * 100
            
            metrics.update({
                'total_return_pct': total_return,
                'annualized_return_pct': annualized_return * 100,
                'annualized_volatility_pct': annualized_volatility * 100,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'win_rate_pct': win_rate
            })
        
        # buy and hold comparison
        trade_start = pd.Timestamp(self.config.trade_start_date)
        price_subset = price_data[price_data.index >= trade_start]
        if not price_subset.empty:
            bh_return = (price_subset['Close'].iloc[-1] / price_subset['Close'].iloc[0] - 1) * 100
            metrics['buy_hold_return_pct'] = bh_return
            metrics['excess_return_pct'] = total_return - bh_return
        
        return metrics

    def make_future_prediction(self, ticker: str, trained_model: 'EnsembleModel' = None,
                              days_ahead: int = 5) -> Dict:
        try:
            logging.info(f"Making future prediction for {ticker}")
            
            # get recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            
            df = yf.download(ticker, start=start_date, end=end_date,
                           interval='1d', auto_adjust=True, progress=False)
            
            if df.empty:
                raise ValueError(f"Could not fetch recent data for {ticker}")
            
            # handle columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # clean and process
            df = df.dropna()
            df = df[df['Volume'] > 0]
            df = self.feature_engineer.create_features(df)
            df = df.dropna()
            
            if len(df) < 50:
                raise ValueError(f"Insufficient recent data for {ticker}")
            
            # use model or retrain
            if trained_model is None:
                if not hasattr(self, 'model') or not self.model.is_fitted:
                    logging.info("Retraining model for future prediction...")
                    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'future_return']
                    feature_cols = [col for col in df.columns if col not in price_cols]
                    
                    X_recent = df[feature_cols].iloc[:-days_ahead].values
                    y_recent = df['future_return'].iloc[:-days_ahead].values
                    
                    # clean data
                    valid_mask = ~(np.isnan(y_recent) | np.isinf(y_recent))
                    for col_idx in range(X_recent.shape[1]):
                        valid_mask &= ~(np.isnan(X_recent[:, col_idx]) | np.isinf(X_recent[:, col_idx]))
                    
                    X_recent = X_recent[valid_mask]
                    y_recent = y_recent[valid_mask]
                    
                    if len(X_recent) < 30:
                        raise ValueError("Insufficient data for retraining")
                    
                    self.model.fit(X_recent, y_recent, feature_cols)
                
                trained_model = self.model
            else:
                trained_model = self.model
            
            # make prediction
            prediction_result = trained_model.predict_future(df, days_ahead)
            
            # add metadata
            prediction_result['ticker'] = ticker
            prediction_result['prediction_date'] = datetime.now().isoformat()
            prediction_result['data_as_of'] = df.index[-1].isoformat()
            prediction_result['total_data_points'] = len(df)
            
            logging.info(f"Future prediction complete for {ticker}")
            logging.info(f"Predicted return: {prediction_result.get('predicted_return_pct', 0):.2f}%")
            logging.info(f"Confidence: {prediction_result.get('confidence_pct', 0):.1f}%")
            logging.info(f"Recommendation: {prediction_result.get('recommendation', 'N/A')}")
            
            return prediction_result
        
        except Exception as e:
            logging.error(f"Future prediction failed for {ticker}: {e}")
            return {
                'ticker': ticker,
                'error': str(e),
                'predicted_return_pct': 0,
                'confidence_pct': 0,
                'recommendation': 'ERROR'
            }

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading_system.log'),
            logging.StreamHandler()
        ]
    )
    
    config = TradingConfig()
    backtester = Backtester(config)
    
    logging.info("ML Trading System")
    logging.info("Starting backtest")
    
    for ticker in config.tickers[:1]:
        try:
            result = backtester.run_backtest(ticker)
            if 'error' not in result:
                metrics = result['metrics']
                logging.info(f"{ticker}: {metrics.get('total_return_pct', 0):.2f}% return")
        except Exception as e:
            logging.error(f"{ticker} failed: {e}")

if __name__ == "__main__":
    main()