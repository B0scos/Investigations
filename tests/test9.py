import numpy as np
import pandas as pd
import scipy.fft as fft
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# ================= CONFIG =================
CSV_PATH = r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv"
PRICE_COL = "close"
DATE_COL = "timestamp"

TRAIN_RATIO = 0.9
ARIMA_ORDER = (1, 0, 1)
GARCH_P, GARCH_Q = 1, 1
EWT_ALPHA = 0.25
NN_LAGS = 5  # number of lagged squared residuals as NN input

# ================= METRICS =================
def qlike(y_true, y_pred_var):
    return np.mean(np.log(y_pred_var) + y_true**2 / y_pred_var)

# ================= DATA =================
def load_returns():
    df = pd.read_csv(CSV_PATH, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL)
    r = np.log(df[PRICE_COL]).diff().dropna()
    return r.values

# ================= EWT FEATURES =================
def ewt_features(x, alpha=EWT_ALPHA):
    """Simple FFT-based EWT-derived features: band variances & ratios"""
    N = len(x)
    X = fft.fft(x)
    freqs = fft.fftfreq(N)

    # Define frequency bands
    cutoff_low = 0.1 * 0.5 * N
    cutoff_mid = 0.3 * 0.5 * N
    cutoff_high = 0.5 * 0.5 * N

    X_low = X.copy()
    X_low[np.abs(freqs) > cutoff_low] = 0
    x_low = np.real(fft.ifft(X_low))

    X_mid = X.copy()
    X_mid[(np.abs(freqs) < cutoff_low) | (np.abs(freqs) > cutoff_mid)] = 0
    x_mid = np.real(fft.ifft(X_mid))

    X_high = X.copy()
    X_high[np.abs(freqs) < cutoff_mid] = 0
    x_high = np.real(fft.ifft(X_high))

    # Compute variance per band
    var_low = np.var(x_low)
    var_mid = np.var(x_mid)
    var_high = np.var(x_high)

    # Ratios
    ratio_hl = var_high / (var_low + 1e-12)
    ratio_ml = var_mid / (var_low + 1e-12)

    return np.array([var_low, var_mid, var_high, ratio_hl, ratio_ml])

# ================= EGARCH + NN HYBRID =================
def forecast_hybrid(train, test):
    # ----------------- ARIMA Mean Model -----------------
    arima = ARIMA(train, order=ARIMA_ORDER).fit()
    resid = train - arima.fittedvalues

    # ----------------- EGARCH -----------------
    garch = arch_model(resid, mean="Zero", vol="EGARCH", p=GARCH_P, q=GARCH_Q, dist="t")
    garch_fit = garch.fit(disp="off")
    H = len(test)
    garch_forecast_var = garch_fit.forecast(horizon=H).variance.values[-1]

    # ----------------- NN Residual Correction -----------------
    # Prepare NN dataset: lagged squared residuals + EWT features
    X_nn, y_nn = [], []
    for t in range(NN_LAGS, len(resid)):
        lagged = resid[t-NN_LAGS:t]**2
        features = ewt_features(resid[t-NN_LAGS:t])
        X_nn.append(np.concatenate([lagged, features]))
        y_nn.append(resid[t]**2)

    X_nn = np.array(X_nn)
    y_nn = np.array(y_nn)

    nn = MLPRegressor(hidden_layer_sizes=(20,), max_iter=500, random_state=42)
    nn.fit(X_nn, y_nn)

    # Forecast NN correction
    nn_correction = []
    last_window = resid[-NN_LAGS:]
    for i in range(H):
        feats = ewt_features(last_window)
        x_in = np.concatenate([last_window**2, feats]).reshape(1, -1)
        pred = nn.predict(x_in)[0]
        nn_correction.append(pred)
        # update last_window
        last_window = np.roll(last_window, -1)
        last_window[-1] = test[i]  # use true value for simplicity (one-step-ahead)

    nn_correction = np.array(nn_correction)

    # Final hybrid forecast
    hybrid_var = garch_forecast_var + nn_correction
    return hybrid_var, garch_forecast_var

# ================= MAIN =================
def main():
    returns = load_returns()
    n = len(returns)
    split = int(n*TRAIN_RATIO)
    train, test = returns[:split], returns[split:]
    H = len(test)

    # ----------------- Forecasts -----------------
    hybrid_var, egarch_var = forecast_hybrid(train, test)

    # Plain GARCH for comparison
    garch_fit = arch_model(train, mean="Zero", vol="GARCH", p=GARCH_P, q=GARCH_Q, dist="t").fit(disp="off")
    garch_var = garch_fit.forecast(horizon=H).variance.values[-1]

    # Plain ARIMA variance proxy
    arima_fcst = ARIMA(train, order=ARIMA_ORDER).fit().forecast(H)
    arima_var = arima_fcst**2

    # ----------------- Evaluation -----------------
    print("\n===== VOLATILITY FORECAST RESULTS =====")
    print("Plain ARIMA (variance proxy):")
    print("  QLIKE :", qlike(test, arima_var))
    print("  MSE   :", mean_squared_error(test**2, arima_var))

    print("\nPlain GARCH:")
    print("  QLIKE :", qlike(test, garch_var))
    print("  MSE   :", mean_squared_error(test**2, garch_var))

    print("\nEGARCH (baseline):")
    print("  QLIKE :", qlike(test, egarch_var))
    print("  MSE   :", mean_squared_error(test**2, egarch_var))

    print("\nHybrid EGARCH + NN (with EWT features):")
    print("  QLIKE :", qlike(test, hybrid_var))
    print("  MSE   :", mean_squared_error(test**2, hybrid_var))

if __name__ == "__main__":
    main()
