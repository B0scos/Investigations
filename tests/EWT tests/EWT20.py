import numpy as np
import pandas as pd
import logging
from numpy.fft import fft, ifft
from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# =========================================================
# EWT FUNCTIONS
# =========================================================

def beta(x):
    x = np.clip(x, 0, 1)
    return x**4 * (35 - 84*x + 70*x**2 - 20*x**3)


def detect_boundaries(signal, n_modes=4):

    N = len(signal)
    spectrum = np.abs(fft(signal))[:N // 2]
    spectrum[0] = 0

    peaks, _ = find_peaks(spectrum)

    if len(peaks) < n_modes:
        idx = np.argsort(spectrum)[-n_modes:]
        peaks = np.sort(idx)

    boundaries = peaks[:n_modes]
    freqs = boundaries / N * np.pi

    return freqs


def build_filterbank(boundaries, N):

    gamma = 0.25
    omega = np.concatenate(([0], boundaries, [np.pi]))

    filters = []

    freqs = np.linspace(0, np.pi, N // 2)

    for i in range(len(omega) - 1):

        w1 = omega[i]
        w2 = omega[i + 1]

        H = np.zeros_like(freqs)
        trans = gamma * (w2 - w1)

        for k, w in enumerate(freqs):

            if w1 + trans <= w <= w2 - trans:
                H[k] = 1

            elif w1 - trans <= w < w1 + trans:
                H[k] = np.sin(np.pi / 2 * beta((w - (w1 - trans)) / (2 * trans)))

            elif w2 - trans < w <= w2 + trans:
                H[k] = np.cos(np.pi / 2 * beta((w - (w2 - trans)) / (2 * trans)))

        full = np.concatenate([H, H[::-1]])
        filters.append(full)

    return filters


def ewt_decompose(signal, n_modes=4):

    N = len(signal)

    boundaries = detect_boundaries(signal, n_modes)
    filters = build_filterbank(boundaries, N)

    fft_sig = fft(signal)

    modes = []

    for f in filters:
        mode = np.real(ifft(fft_sig * f))
        modes.append(mode)

    return modes


# =========================================================
# HAR-RV MODEL
# =========================================================

def har_features(rv):

    daily = rv[:-1]
    weekly = pd.Series(rv).rolling(5).mean().values[:-1]
    monthly = pd.Series(rv).rolling(22).mean().values[:-1]

    X = np.column_stack([daily, weekly, monthly])
    y = rv[1:]

    valid = ~np.isnan(X).any(axis=1)

    return X[valid], y[valid]


# =========================================================
# LOAD BTC DATA
# =========================================================

PATH = r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv"

df = pd.read_csv(PATH)

price = df["close"].values

ret = np.diff(np.log(price))
rv = ret**2

# =========================================================
# SPLIT
# =========================================================

split = int(len(rv) * 0.7)

train = rv[:split]
test = rv[split:]

logging.info(f"Train size: {len(train)}")
logging.info(f"Test size : {len(test)}")

# =========================================================
# ===== PERSISTENCE BENCHMARK =====
# =========================================================

pred_persist = train[-1] * np.ones_like(test)

# =========================================================
# ===== HAR BENCHMARK =====
# =========================================================

X_train, y_train = har_features(train)

har = LinearRegression()
har.fit(X_train, y_train)

X_test, _ = har_features(np.concatenate([train[-22:], test]))

pred_har = har.predict(X_test)

pred_har = pred_har[:len(test)]

# =========================================================
# ===== EWT + MODE ARIMA =====
# =========================================================

logging.info("Running EWT...")

modes_train = ewt_decompose(train, n_modes=4)

# DENOISE — REMOVE LAST MODE
modes_train = modes_train[:-1]

pred_modes = []

for i, mode in enumerate(modes_train):

    logging.info(f"ARIMA on mode {i+1}")

    model = ARIMA(mode, order=(1,0,1))
    fit = model.fit()

    pred = fit.forecast(len(test))
    pred_modes.append(pred)

pred_ewt = np.sum(pred_modes, axis=0)

# =========================================================
# METRICS
# =========================================================

def report(name, y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    logging.info(f"{name:<12} | MSE={mse*100:.6f} | MAE={mae*100:.6f}")

    return mse, mae


logging.info("====== OOS RESULTS ======")

mse_p, mae_p = report("Persistence", test, pred_persist)
mse_h, mae_h = report("HAR", test, pred_har)
mse_e, mae_e = report("EWT+ARIMA", test, pred_ewt)

logging.info("Pipeline finished successfully.")
