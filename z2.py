import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from arch import arch_model
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, classification_report
import warnings

warnings.filterwarnings("ignore")  # optional, keeps output clean

# ---------------------------
# 1. Fetch BTC data
# ---------------------------
def fetch_ohlcv(symbol='BTC/USD', timeframe='5m', since_days=365*3):
    kraken = ccxt.kraken()
    since = kraken.parse8601((datetime.utcnow() - timedelta(days=since_days)).strftime('%Y-%m-%dT%H:%M:%S'))
    all_data = []
    while True:
        data = kraken.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not data:
            break
        all_data += data
        since = data[-1][0] + 1
        if len(data) < 1000:
            break
    df = pd.DataFrame(all_data, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# ---------------------------
# 2. Compute daily realized variance
# ---------------------------
def compute_daily_rv(df):
    df = df.copy()
    df['log_return'] = np.log(df['close']).diff()
    df['date'] = df['timestamp'].dt.date
    rv = df.groupby('date')['log_return'].apply(lambda x: np.sum(x**2))
    return rv

# ---------------------------
# 3. Prepare features with directional targets
# ---------------------------
def prepare_features_with_direction(rv):
    """
    Prepare features with added directional targets for volatility
    """
    df = pd.DataFrame({'RV_d': rv})
    
    # Basic features
    df['RV_w'] = df['RV_d'].rolling(7).mean()
    df['RV_m'] = df['RV_d'].rolling(30).mean()
    
    # Lagged features
    df['RV_d_lag1'] = df['RV_d'].shift(1)
    df['RV_w_lag1'] = df['RV_w'].shift(1)
    df['RV_m_lag1'] = df['RV_m'].shift(1)
    
    # Volatility change features
    df['RV_d_change'] = df['RV_d'].pct_change()
    df['RV_w_change'] = df['RV_w'].pct_change()
    df['RV_m_change'] = df['RV_m'].pct_change()
    
    # Technical indicators for volatility direction
    df['RV_d_ma5'] = df['RV_d'].rolling(5).mean()
    df['RV_d_ma10'] = df['RV_d'].rolling(10).mean()
    df['RV_d_std5'] = df['RV_d'].rolling(5).std()
    df['RV_d_std10'] = df['RV_d'].rolling(10).std()
    
    # Directional targets
    # Target 1: Binary direction (up/down)
    df['direction_target'] = (df['RV_d'] > df['RV_d'].shift(1)).astype(int)
    
    # Target 2: Magnitude of change
    df['magnitude_target'] = df['RV_d_change']
    
    # Target 3: Categorical direction (high/low/medium)
    percentile_33 = df['RV_d_change'].quantile(0.33)
    percentile_66 = df['RV_d_change'].quantile(0.66)
    df['category_target'] = pd.cut(df['RV_d_change'], 
                                   bins=[-np.inf, percentile_33, percentile_66, np.inf],
                                   labels=[0, 1, 2])  # 0=low, 1=medium, 2=high
    
    # Lag the targets for prediction
    df['direction_target'] = df['direction_target'].shift(-1)
    df['magnitude_target'] = df['magnitude_target'].shift(-1)
    df['category_target'] = df['category_target'].shift(-1)
    
    df = df.dropna()
    
    # Features for direction prediction
    direction_features = ['RV_d_lag1', 'RV_w_lag1', 'RV_m_lag1',
                         'RV_d_change', 'RV_w_change', 'RV_m_change',
                         'RV_d_ma5', 'RV_d_ma10', 'RV_d_std5', 'RV_d_std10']
    
    X = df[direction_features].astype(float)
    y_direction = df['direction_target'].astype(int)
    y_magnitude = df['magnitude_target'].astype(float)
    y_category = df['category_target'].astype(int)
    y_rv = df['RV_d'].astype(float)  # Original RV for level prediction
    
    return X, y_rv, y_direction, y_magnitude, y_category

# ---------------------------
# 4. Base model forecasts
# ---------------------------
def fit_garch(y_train):
    """GARCH(1,1) forecast with automatic fallback"""
    if len(y_train) < 10 or np.all(y_train == 0):
        return np.var(y_train)  # fallback for very small samples or zero variance
    try:
        am = arch_model(y_train, vol='Garch', p=1, q=1, dist='normal')
        res = am.fit(disp='off', show_warning=False)
        forecast = res.forecast(horizon=1).variance.values[-1,0]
    except Exception as e:
        # if optimizer fails, use variance of training series
        forecast = np.var(y_train)
    return forecast

def base_forecasts(X_train, y_train, X_pred):
    """Generate point forecasts for Ridge, HAR, GARCH"""
    # Ridge regression
    ridge = Ridge().fit(X_train, y_train)
    y_hat_ridge = ridge.predict(X_pred)
    
    # HAR approximation (using mean of features as in paper)
    y_hat_har = X_pred.mean(axis=1).values
    
    # GARCH(1,1) forecasts
    y_hat_garch = np.array([fit_garch(y_train) for _ in range(len(X_pred))])
    
    return pd.DataFrame({
        'ridge': y_hat_ridge,
        'har': y_hat_har,
        'garch': y_hat_garch
    })

# ---------------------------
# 5. Directional Volatility Models
# ---------------------------
class DirectionalVolatilityPredictor:
    """Predict direction and magnitude of volatility changes"""
    
    def __init__(self):
        self.direction_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.magnitude_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.category_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def fit(self, X_train, y_direction_train, y_magnitude_train, y_category_train):
        """Train all directional models"""
        self.direction_model.fit(X_train, y_direction_train)
        self.magnitude_model.fit(X_train, y_magnitude_train)
        self.category_model.fit(X_train, y_category_train)
    
    def predict(self, X_test):
        """Make directional predictions"""
        direction_pred = self.direction_model.predict(X_test)
        direction_proba = self.direction_model.predict_proba(X_test)[:, 1]  # Probability of increase
        
        magnitude_pred = self.magnitude_model.predict(X_test)
        category_pred = self.category_model.predict(X_test)
        
        return {
            'direction': direction_pred,  # 0=down, 1=up
            'direction_proba': direction_proba,  # probability of increase
            'magnitude': magnitude_pred,  # percentage change
            'category': category_pred  # 0=low, 1=medium, 2=high
        }
    
    def evaluate(self, X_test, y_direction_test, y_magnitude_test, y_category_test):
        """Evaluate directional predictions"""
        predictions = self.predict(X_test)
        
        print("\n" + "="*60)
        print("DIRECTIONAL VOLATILITY PREDICTION EVALUATION")
        print("="*60)
        
        # Direction accuracy
        direction_acc = accuracy_score(y_direction_test, predictions['direction'])
        direction_f1 = f1_score(y_direction_test, predictions['direction'])
        
        print(f"\nDirection Prediction (Up/Down):")
        print(f"  Accuracy: {direction_acc:.4f}")
        print(f"  F1 Score: {direction_f1:.4f}")
        
        # Magnitude prediction
        magnitude_mae = np.mean(np.abs(y_magnitude_test - predictions['magnitude']))
        magnitude_mse = np.mean((y_magnitude_test - predictions['magnitude'])**2)
        
        print(f"\nMagnitude Prediction (Percentage Change):")
        print(f"  MAE: {magnitude_mae:.6f}")
        print(f"  MSE: {magnitude_mse:.8f}")
        
        # Category prediction
        category_acc = accuracy_score(y_category_test, predictions['category'])
        print(f"\nCategory Prediction (Low/Medium/High):")
        print(f"  Accuracy: {category_acc:.4f}")
        
        # Detailed classification report
        print("\nDetailed Classification Report (Direction):")
        print(classification_report(y_direction_test, predictions['direction'], 
                                   target_names=['Vol Down', 'Vol Up']))
        
        # Trading strategy simulation
        self.simulate_trading_strategy(predictions, y_direction_test, y_magnitude_test)
        
        return {
            'direction_accuracy': direction_acc,
            'direction_f1': direction_f1,
            'magnitude_mae': magnitude_mae,
            'magnitude_mse': magnitude_mse,
            'category_accuracy': category_acc
        }
    
    def simulate_trading_strategy(self, predictions, y_direction_test, y_magnitude_test):
        """Simulate a simple trading strategy based on volatility direction"""
        # Simple strategy: Long volatility when predicted to increase
        signals = predictions['direction']  # 1 = buy signal
        
        # Calculate returns (using magnitude as proxy for volatility trading P&L)
        # In reality, this would be VIX futures or volatility ETFs
        strategy_returns = []
        buy_and_hold_returns = []
        
        for i in range(len(signals)):
            if signals[i] == 1:  # Buy signal (volatility increase expected)
                # Simulate return based on actual volatility change
                # Positive when correctly predicting increase, negative when wrong
                if y_direction_test.iloc[i] == 1:  # Actually increased
                    ret = abs(y_magnitude_test.iloc[i])  # Gain proportional to magnitude
                else:  # Actually decreased
                    ret = -abs(y_magnitude_test.iloc[i])  # Loss proportional to magnitude
            else:  # No position
                ret = 0
            
            strategy_returns.append(ret)
            buy_and_hold_returns.append(abs(y_magnitude_test.iloc[i]))  # Always long
        
        strategy_returns = np.array(strategy_returns)
        buy_and_hold_returns = np.array(buy_and_hold_returns)
        
        # Calculate statistics
        total_return = np.sum(strategy_returns)
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0
        win_rate = np.mean(strategy_returns > 0) * 100
        
        print(f"\nTrading Strategy Simulation:")
        print(f"  Total Return: {total_return:.4f}")
        print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Number of Trades: {np.sum(signals)}")
        
        # Compare to buy-and-hold
        bh_total = np.sum(buy_and_hold_returns)
        bh_sharpe = np.mean(buy_and_hold_returns) / np.std(buy_and_hold_returns) if np.std(buy_and_hold_returns) > 0 else 0
        
        print(f"\nBuy-and-Hold (Always Long Volatility):")
        print(f"  Total Return: {bh_total:.4f}")
        print(f"  Sharpe Ratio: {bh_sharpe:.4f}")
        
        return {
            'strategy_return': total_return,
            'strategy_sharpe': sharpe_ratio,
            'strategy_win_rate': win_rate,
            'bh_return': bh_total,
            'bh_sharpe': bh_sharpe
        }

# ---------------------------
# 6. Combined Volatility & Direction Prediction
# ---------------------------
def combined_volatility_prediction(X_train, y_rv_train, y_dir_train, y_mag_train, y_cat_train,
                                  X_test, y_rv_test, y_dir_test, y_mag_test, y_cat_test):
    """
    Combine level prediction with directional prediction
    """
    print("\n" + "="*60)
    print("COMBINED VOLATILITY FORECASTING SYSTEM")
    print("="*60)
    
    # 1. Level prediction (existing models)
    print("\n1. Volatility Level Prediction:")
    y_hat_train = base_forecasts(X_train, y_rv_train, X_train)
    y_hat_test = base_forecasts(X_train, y_rv_train, X_test)
    
    # 2. Directional prediction
    print("\n2. Volatility Direction Prediction:")
    dir_predictor = DirectionalVolatilityPredictor()
    dir_predictor.fit(X_train, y_dir_train, y_mag_train, y_cat_train)
    dir_results = dir_predictor.evaluate(X_test, y_dir_test, y_mag_test, y_cat_test)
    
    # 3. Combine predictions
    print("\n3. Combined Predictions Analysis:")
    predictions = dir_predictor.predict(X_test)
    
    # Create enhanced forecasts using both level and direction
    enhanced_forecasts = []
    for i in range(len(X_test)):
        base_pred = y_hat_test.iloc[i]['har']  # Use HAR as base
        
        # Adjust based on direction and magnitude
        if predictions['direction'][i] == 1:  # Predicted increase
            adjustment = abs(predictions['magnitude'][i]) * base_pred
            enhanced = base_pred + adjustment
        else:  # Predicted decrease
            adjustment = abs(predictions['magnitude'][i]) * base_pred
            enhanced = base_pred - adjustment
        
        enhanced_forecasts.append(enhanced)
    
    enhanced_forecasts = np.array(enhanced_forecasts)
    
    # Evaluate enhanced forecasts
    enhanced_mae = np.mean(np.abs(y_rv_test - enhanced_forecasts))
    enhanced_mse = np.mean((y_rv_test - enhanced_forecasts)**2)
    enhanced_rmse = np.sqrt(enhanced_mse)
    
    print(f"\nEnhanced Forecasts (Level + Direction):")
    print(f"  MAE: {enhanced_mae:.8f}")
    print(f"  MSE: {enhanced_mse:.10f}")
    print(f"  RMSE: {enhanced_rmse:.8f}")
    
    # Compare to base HAR model
    har_mae = np.mean(np.abs(y_rv_test - y_hat_test['har']))
    har_rmse = np.sqrt(np.mean((y_rv_test - y_hat_test['har'])**2))
    
    print(f"\nComparison with Base HAR Model:")
    print(f"  HAR MAE: {har_mae:.8f}")
    print(f"  HAR RMSE: {har_rmse:.8f}")
    print(f"  Improvement in MAE: {(har_mae - enhanced_mae)/har_mae*100:.2f}%")
    
    # Generate trading signals based on combined prediction
    generate_trading_signals(y_hat_test, predictions, y_rv_test, y_dir_test)
    
    return {
        'enhanced_forecasts': enhanced_forecasts,
        'enhanced_mae': enhanced_mae,
        'enhanced_rmse': enhanced_rmse,
        'direction_results': dir_results
    }

def generate_trading_signals(y_hat_test, direction_predictions, y_rv_test, y_dir_test):
    """Generate practical trading signals"""
    print("\n" + "="*60)
    print("TRADING SIGNAL GENERATION")
    print("="*60)
    
    signals = []
    confidence_scores = []
    
    for i in range(len(y_hat_test)):
        base_vol = y_hat_test.iloc[i]['har']
        direction = direction_predictions['direction'][i]
        proba = direction_predictions['direction_proba'][i]
        
        # Generate signal based on strength of prediction
        if direction == 1 and proba > 0.6:  # Strong buy signal
            signal = 2  # Strong buy
        elif direction == 1 and proba > 0.5:  # Weak buy signal
            signal = 1  # Weak buy
        elif direction == 0 and proba < 0.4:  # Strong sell signal
            signal = -2  # Strong sell
        elif direction == 0 and proba < 0.5:  # Weak sell signal
            signal = -1  # Weak sell
        else:
            signal = 0  # Hold
        
        signals.append(signal)
        confidence_scores.append(abs(proba - 0.5) * 2)  # Convert to 0-1 confidence
    
    signals = np.array(signals)
    confidence_scores = np.array(confidence_scores)
    
    # Analyze signal performance
    correct_signals = 0
    total_signals = 0
    
    for i in range(len(signals)):
        if signals[i] != 0:  # Only count when we take a position
            total_signals += 1
            if (signals[i] > 0 and y_dir_test.iloc[i] == 1) or (signals[i] < 0 and y_dir_test.iloc[i] == 0):
                correct_signals += 1
    
    if total_signals > 0:
        signal_accuracy = correct_signals / total_signals * 100
    else:
        signal_accuracy = 0
    
    print(f"\nSignal Statistics:")
    print(f"  Total Signals Generated: {total_signals}")
    print(f"  Strong Buy Signals: {np.sum(signals == 2)}")
    print(f"  Weak Buy Signals: {np.sum(signals == 1)}")
    print(f"  Strong Sell Signals: {np.sum(signals == -2)}")
    print(f"  Weak Sell Signals: {np.sum(signals == -1)}")
    print(f"  Hold Signals: {np.sum(signals == 0)}")
    print(f"  Signal Accuracy: {signal_accuracy:.2f}%")
    
    # Calculate signal confidence statistics
    if total_signals > 0:
        signal_indices = np.where(signals != 0)[0]
        avg_confidence = np.mean(confidence_scores[signal_indices]) * 100
        print(f"  Average Signal Confidence: {avg_confidence:.2f}%")
    
    # Show recent signals
    print(f"\nRecent Trading Signals (last 10 periods):")
    recent_signals = signals[-10:] if len(signals) >= 10 else signals
    recent_confidences = confidence_scores[-10:] if len(confidence_scores) >= 10 else confidence_scores
    
    for i in range(len(recent_signals)):
        signal_map = {2: 'STRONG BUY', 1: 'WEAK BUY', 0: 'HOLD', -1: 'WEAK SELL', -2: 'STRONG SELL'}
        signal_text = signal_map.get(recent_signals[i], 'UNKNOWN')
        print(f"  Period {i+1}: {signal_text} (Confidence: {recent_confidences[i]*100:.1f}%)")

# ---------------------------
# 7. Quantile Forecasting Methods (Existing)
# ---------------------------
def qrs(y_hat_train, y_train, y_hat_test, quantiles=[0.01,0.05,0.5,0.95,0.99]):
    residuals = y_train.values - y_hat_train.mean(axis=1).values
    simulated = y_hat_test.mean(axis=1).values[:, None] + residuals
    quantile_preds = np.percentile(simulated, [q*100 for q in quantiles], axis=1).T
    return pd.DataFrame(quantile_preds, columns=[f'q{int(q*100)}' for q in quantiles])

def qlr(y_hat_train, y_train, y_hat_test, quantiles=[0.01,0.05,0.5,0.95,0.99]):
    y_train = y_train.reset_index(drop=True).astype(float)
    y_hat_train = y_hat_train.reset_index(drop=True).astype(float)
    y_hat_test = y_hat_test.reset_index(drop=True).astype(float)
    
    preds = []
    for q in quantiles:
        model = QuantReg(y_train, y_hat_train).fit(q=q)
        preds.append(model.predict(y_hat_test))
    return pd.DataFrame(np.column_stack(preds), columns=[f'q{int(q*100)}' for q in quantiles])

def qrf(y_hat_train, y_train, y_hat_test, quantiles=[0.01,0.05,0.5,0.95,0.99]):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(y_hat_train, y_train)
    preds = []
    for q in quantiles:
        leaf_values = []
        for tree in rf.estimators_:
            leaf_nodes_test = tree.apply(y_hat_test)
            leaf_nodes_train = tree.apply(y_hat_train)
            values = np.array([y_train.values[leaf_nodes_train == ln].mean() if np.any(leaf_nodes_train==ln) else 0 
                               for ln in leaf_nodes_test])
            leaf_values.append(values)
        leaf_values = np.array(leaf_values)
        preds.append(np.percentile(leaf_values, q*100, axis=0))
    return pd.DataFrame(np.column_stack(preds), columns=[f'q{int(q*100)}' for q in quantiles])

# ---------------------------
# 8. Main pipeline with directional prediction
# ---------------------------
if __name__ == "__main__":
    # Use your data loading method
    df = pd.read_csv('data/BTC_data_hour.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    
    print("Computing realized variance...")
    rv = compute_daily_rv(df)
    
    print("Preparing features with directional targets...")
    X, y_rv, y_direction, y_magnitude, y_category = prepare_features_with_direction(rv)
    
    if len(X) < 50:
        raise ValueError(f"Not enough data to train models. Samples: {len(X)}")
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_rv_train, y_rv_test, y_dir_train, y_dir_test, y_mag_train, y_mag_test, y_cat_train, y_cat_test = train_test_split(
        X, y_rv, y_direction, y_magnitude, y_category, test_size=0.2, shuffle=False
    )
    
    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples")
    
    # Run combined volatility and direction prediction
    combined_results = combined_volatility_prediction(
        X_train, y_rv_train, y_dir_train, y_mag_train, y_cat_train,
        X_test, y_rv_test, y_dir_test, y_mag_test, y_cat_test
    )
    
    # Continue with probabilistic forecasting
    print("\n" + "="*60)
    print("PROBABILISTIC FORECASTING WITH DIRECTIONAL INSIGHTS")
    print("="*60)
    
    # Generate base model forecasts
    y_hat_train = base_forecasts(X_train, y_rv_train, X_train)
    y_hat_test = base_forecasts(X_train, y_rv_train, X_test)
    
    # Generate probabilistic forecasts
    quantile_levels = [0.01, 0.05, 0.5, 0.95, 0.99]
    
    qrs_preds = qrs(y_hat_train, y_rv_train, y_hat_test, quantile_levels)
    qlr_preds = qlr(y_hat_train, y_rv_train, y_hat_test, quantile_levels)
    qrf_preds = qrf(y_hat_train, y_rv_train, y_hat_test, quantile_levels)
    
    print("\nAnalysis complete! Summary of capabilities:")
    print("✓ Volatility level prediction (traditional)")
    print("✓ Volatility direction prediction (up/down)")
    print("✓ Volatility magnitude prediction (how much change)")
    print("✓ Volatility category prediction (low/medium/high)")
    print("✓ Probabilistic forecasting with quantiles")
    print("✓ Trading signal generation")
    print("✓ Strategy simulation and backtesting")
    
    print("\nNext steps:")
    print("1. Review directional accuracy metrics above")
    print("2. Check trading signal accuracy")
    print("3. Consider implementing volatility derivatives trading strategy")
    print("4. Adjust model parameters based on performance")