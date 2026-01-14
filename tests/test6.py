import numpy as np
import pandas as pd
import scipy.fft as fft
from scipy.signal import find_peaks
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# ================= CONFIG =================
CSV_PATH = r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv"
PRICE_COL = "close"
DATE_COL = "timestamp"

TRAIN_RATIO = 0.9
EWT_ALPHA = 0.25
ARIMA_ORDER = (1, 0, 1)
GARCH_P, GARCH_Q = 1, 1
MLP_HIDDEN = 10

# ================= METRIC =================
def rmse(y, yhat):
    return np.sqrt(mean_squared_error(y, yhat))

# ================= DATA =================
def load_returns():
    df = pd.read_csv(CSV_PATH, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL)
    r = np.log(df[PRICE_COL]).diff().dropna()
    return r.values

# ================= EWT =================
def ewt_decompose(x, alpha=0.25, min_peaks=3):
    N = len(x)
    X = np.abs(fft.fft(x))
    freqs = fft.fftfreq(N)

    mask = freqs > 0
    X, freqs = X[mask], freqs[mask]

    Xn = (X - X.min()) / (X.max() - X.min() + 1e-12)
    peaks, _ = find_peaks(Xn, height=alpha)
    if len(peaks) < min_peaks:
        peaks = np.argsort(Xn)[-min_peaks:]
    peak_freqs = np.sort(freqs[peaks])
    bounds = 0.5 * (peak_freqs[:-1] + peak_freqs[1:])
    bounds = np.pi * bounds / bounds.max()
    return build_ewt_bands(x, bounds)

def build_ewt_bands(x, bounds):
    N = len(x)
    X = fft.fft(x)
    freqs = np.linspace(0, np.pi, N // 2 + 1)

    def mirror(H):
        out = np.zeros(N)
        out[:len(H)] = H
        out[len(H):] = H[-2:0:-1]
        return out

    bands = []

    # Low-pass
    H = np.zeros_like(freqs)
    H[freqs <= bounds[0]] = 1
    bands.append(np.real(fft.ifft(X * mirror(H))))

    # Band-pass
    for i in range(len(bounds) - 1):
        H = np.zeros_like(freqs)
        H[(freqs >= bounds[i]) & (freqs <= bounds[i + 1])] = 1
        bands.append(np.real(fft.ifft(X * mirror(H))))

    # High-pass
    H = np.zeros_like(freqs)
    H[freqs >= bounds[-1]] = 1
    bands.append(np.real(fft.ifft(X * mirror(H))))

    return bands

# ================= FORECAST PER BAND =================
def forecast_band(band_train, horizon):
    # ARIMA for linear part
    arima = ARIMA(band_train, order=ARIMA_ORDER).fit()
    arima_fcst = arima.forecast(horizon)

    # MLP on residuals for nonlinear part
    resid = band_train - arima.fittedvalues
    X = np.arange(len(resid)).reshape(-1,1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    mlp = MLPRegressor(hidden_layer_sizes=(MLP_HIDDEN,), max_iter=2000)
    mlp.fit(X_scaled, resid)
    X_forecast = np.arange(len(resid), len(resid)+horizon).reshape(-1,1)
    Xf_scaled = scaler.transform(X_forecast)
    mlp_fcst = mlp.predict(Xf_scaled)

    return arima_fcst + mlp_fcst

# ================= MAIN =================
def main():
    returns = load_returns()
    n = len(returns)
    split = int(n*TRAIN_RATIO)
    train, test = returns[:split], returns[split:]
    H = len(test)

    # ------------------ BASELINE: ARIMA + GARCH ------------------
    base = arch_model(train, mean="AR", lags=1, vol="GARCH",
                      p=GARCH_P, q=GARCH_Q, dist="t")
    base_fit = base.fit(disp="off")
    base_var = base_fit.forecast(horizon=H).mean.values[-1]

    # ------------------ HYBRID: EWT + ARIMA + MLP ------------------
    bands = ewt_decompose(train, alpha=EWT_ALPHA)
    hybrid_fcst = np.zeros(H)
    for b in bands:
        hybrid_fcst += forecast_band(b, H)

    # ------------------ RESULTS ------------------
    print("\n===== RESULTS =====")
    print("Baseline ARIMA-GARCH RMSE :", rmse(test, base_var))
    print("EWT Hybrid RMSE           :", rmse(test, hybrid_fcst))

if __name__ == "__main__":
    main()
