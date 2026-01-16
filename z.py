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
    
    # HAR approximation
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
# 6. RMSE Evaluation
# ---------------------------
def evaluate_rmse(y_test, y_hat_test, qrs_preds=None, qlr_preds=None, qrf_preds=None):
    print("=== Base Models RMSE ===")
    for col in y_hat_test.columns:
        rmse = np.sqrt(mean_squared_error(y_test, y_hat_test[col]))
        print(f"{col}: {rmse:.8f}")

    # Optional: quantile forecasts
    if qrs_preds is not None:
        print("\n=== Quantile Forecasts RMSE (median q50) ===")
        for name, preds in zip(['QRS','QLR','QRF'], [qrs_preds, qlr_preds, qrf_preds]):
            if preds is not None and 'q50' in preds.columns:
                rmse = np.sqrt(mean_squared_error(y_test, preds['q50']))
                print(f"{name} median: {rmse:.8f}")

# ---------------------------
# 7. Main pipeline
# ---------------------------
if __name__ == "__main__":
    df = pd.read_csv('data/BTC_data_hour.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    rv = compute_daily_rv(df)
    X, y = prepare_features(rv)
    
    if len(X) < 50:
        raise ValueError(f"Not enough data to train models. Samples: {len(X)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    y_hat_train = base_forecasts(X_train, y_train, X_train)
    y_hat_test = base_forecasts(X_train, y_train, X_test)
    
    qrs_preds = qrs(y_hat_train, y_train, y_hat_test)
    qlr_preds = qlr(y_hat_train, y_train, y_hat_test)
    qrf_preds = qrf(y_hat_train, y_train, y_hat_test)
    
    evaluate_rmse(y_test, y_hat_test, qrs_preds, qlr_preds, qrf_preds)
