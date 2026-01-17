import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from arch import arch_model
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.metrics import mean_squared_error
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
# 3. Prepare features
# ---------------------------
def prepare_features(rv):
    df = pd.DataFrame({'RV_d': rv})
    df['RV_w'] = df['RV_d'].rolling(7).mean()
    df['RV_m'] = df['RV_d'].rolling(30).mean()
    df['RV_d_lag1'] = df['RV_d'].shift(1)
    df['RV_w_lag1'] = df['RV_w'].shift(1)
    df['RV_m_lag1'] = df['RV_m'].shift(1)
    df = df.dropna()
    X = df[['RV_d_lag1','RV_w_lag1','RV_m_lag1']].astype(float)
    y = df['RV_d'].astype(float)
    return X, y

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
# 5. Quantile Forecasting Methods
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
# 6. Simple Direction Win Rate
# ---------------------------
def calculate_direction_win_rate(y_true, y_pred, model_name=""):
    """
    Simple win rate for volatility direction prediction
    Compares if predicted direction (up/down) matches actual direction
    """
    # Calculate actual directions: 1 if current > previous (up), 0 if down
    actual_directions = (y_true[1:].values > y_true[:-1].values).astype(int)
    
    # Calculate predicted directions: using y_pred[1:] vs y_true[:-1]
    predicted_directions = (y_pred[1:] > y_true[:-1].values).astype(int)
    
    # Calculate win rate
    correct_predictions = (predicted_directions == actual_directions)
    win_rate = np.mean(correct_predictions) * 100
    
    return win_rate, len(correct_predictions), np.sum(correct_predictions)

# ---------------------------
# 7. Evaluation Functions
# ---------------------------
def evaluate_base_models(y_test, y_hat_test):
    """Traditional RMSE evaluation for base models"""
    print("\n" + "="*60)
    print("BASE MODELS RMSE EVALUATION")
    print("="*60)
    
    rmse_results = {}
    for col in y_hat_test.columns:
        rmse = np.sqrt(mean_squared_error(y_test, y_hat_test[col]))
        rmse_results[col] = rmse
        print(f"{col}: {rmse:.8f}")
    
    return rmse_results

def evaluate_probabilistic(y_test, qrs_preds=None, qlr_preds=None, qrf_preds=None):
    """Probabilistic forecast evaluation"""
    print("\n" + "="*60)
    print("PROBABILISTIC FORECAST EVALUATION METRICS")
    print("="*60)
    
    # Evaluate QRS if available
    if qrs_preds is not None and 'q50' in qrs_preds.columns:
        win_rate, total, correct = calculate_direction_win_rate(y_test, qrs_preds['q50'])
        print(f"\nQRS (Median q50):")
        print(f"  Direction Win Rate: {win_rate:.2f}% ({correct}/{total})")
    
    # Evaluate QLR if available
    if qlr_preds is not None and 'q50' in qlr_preds.columns:
        win_rate, total, correct = calculate_direction_win_rate(y_test, qlr_preds['q50'])
        print(f"\nQLR (Median q50):")
        print(f"  Direction Win Rate: {win_rate:.2f}% ({correct}/{total})")
    
    # Evaluate QRF if available
    if qrf_preds is not None and 'q50' in qrf_preds.columns:
        win_rate, total, correct = calculate_direction_win_rate(y_test, qrf_preds['q50'])
        print(f"\nQRF (Median q50):")
        print(f"  Direction Win Rate: {win_rate:.2f}% ({correct}/{total})")

# ---------------------------
# 8. Main pipeline
# ---------------------------
if __name__ == "__main__":
    # Use your data loading method
    df = pd.read_csv('data/BTC_data_hour.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    
    print("Computing realized variance...")
    rv = compute_daily_rv(df)
    
    print("Preparing features...")
    X, y = prepare_features(rv)
    
    if len(X) < 50:
        raise ValueError(f"Not enough data to train models. Samples: {len(X)}")
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples")
    
    # Generate base model forecasts
    print("\nGenerating base model forecasts...")
    y_hat_train = base_forecasts(X_train, y_train, X_train)
    y_hat_test = base_forecasts(X_train, y_train, X_test)
    
    # Generate probabilistic forecasts
    print("\nGenerating probabilistic forecasts...")
    quantile_levels = [0.01, 0.05, 0.5, 0.95, 0.99]
    
    qrs_preds = qrs(y_hat_train, y_train, y_hat_test, quantile_levels)
    qlr_preds = qlr(y_hat_train, y_train, y_hat_test, quantile_levels)
    qrf_preds = qrf(y_hat_train, y_train, y_hat_test, quantile_levels)
    
    # Evaluate base models with direction win rate
    print("\n" + "="*60)
    print("DIRECTION WIN RATES - BASE MODELS")
    print("="*60)
    
    for model_name in y_hat_test.columns:
        win_rate, total, correct = calculate_direction_win_rate(y_test, y_hat_test[model_name])
        print(f"\n{model_name.upper()}:")
        print(f"  Direction Win Rate: {win_rate:.2f}% ({correct}/{total})")
    
    # Evaluate probabilistic models
    evaluate_probabilistic(y_test, qrs_preds, qlr_preds, qrf_preds)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nDone! Only direction win rates added as requested.")