# ============================================================
# IMPROVED EWT + ARIMA + EGARCH-t VOLATILITY BENCHMARK
# ============================================================

import numpy as np
import pandas as pd
from numpy.fft import fft, ifft, fftfreq

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv" 
PRICE_COL = "close"
DATE_COL = "timestamp"

TRAIN_RATIO = 0.8
EWT_ALPHA = 0.05
EWT_K = 2                 # number of low-frequency modes
ARIMA_ORDER = (1, 0, 1)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL)

prices = df[PRICE_COL].values
returns = np.diff(np.log(prices))
returns = returns[~np.isnan(returns)]

train_size = int(len(returns) * TRAIN_RATIO)

# ============================================================
# EWT IMPLEMENTATION
# ============================================================

def ewt_decompose(signal, alpha=0.05):
    N = len(signal)

    X = np.abs(fft(signal))[:N // 2]
    freqs = fftfreq(N)[:N // 2]

    Xn = (X - X.min()) / (X.max() - X.min())
    at = Xn.min() + alpha * (Xn.max() - Xn.min())

    J = freqs[Xn >= at]

    if len(J) < 2:
        return [signal]

    boundaries = (J[:-1] + J[1:]) / 2
    boundaries = np.concatenate(([0], boundaries, [0.5]))

    omega = np.abs(fftfreq(N))
    Xf = fft(signal)

    modes = []
    for k in range(len(boundaries) - 1):
        mask = (omega >= boundaries[k]) & (omega <= boundaries[k + 1])
        modes.append(np.real(ifft(Xf * mask)))

    return modes


def ewt_low_freq(signal, alpha=0.05, k=2):
    modes = ewt_decompose(signal, alpha)
    k = min(k, len(modes))
    return np.sum(modes[:k], axis=0)

# ============================================================
# MODELS
# ============================================================

def egarch_t_forecast(returns, train_size):
    f, y = [], []

    for t in range(train_size, len(returns) - 1):
        model = arch_model(
            returns[:t],
            vol="EGARCH",
            p=1,
            q=1,
            mean="Zero",
            dist="t",
            rescale=False
        )
        res = model.fit(disp="off")

        f.append(res.forecast(horizon=1).variance.iloc[-1, 0])
        y.append(returns[t + 1] ** 2)

    return np.array(f), np.array(y)


def ewt_egarch_forecast(returns, train_size, alpha, k):
    signal = ewt_low_freq(returns, alpha, k)
    f, y = [], []

    for t in range(train_size, len(signal) - 1):
        model = arch_model(
            signal[:t],
            vol="EGARCH",
            p=1,
            q=1,
            mean="Zero",
            dist="t",
            rescale=False
        )
        res = model.fit(disp="off")

        f.append(res.forecast(horizon=1).variance.iloc[-1, 0])
        y.append(signal[t + 1] ** 2)

    return np.array(f), np.array(y)


def ewt_arima_egarch_forecast(returns, train_size, order, alpha, k):
    signal = ewt_low_freq(returns, alpha, k)
    f, y = [], []

    for t in range(train_size, len(signal) - 1):
        arima = ARIMA(signal[:t], order=order).fit()
        resid = arima.resid

        garch = arch_model(
            resid,
            vol="EGARCH",
            p=1,
            q=1,
            mean="Zero",
            dist="t",
            rescale=False
        )
        res = garch.fit(disp="off")

        f.append(res.forecast(horizon=1).variance.iloc[-1, 0])
        y.append(signal[t + 1] ** 2)

    return np.array(f), np.array(y)

# ============================================================
# EVALUATION
# ============================================================

def evaluate(name, f, y):
    mse = mean_squared_error(y, f)
    print(f"{name:<35} MSE = {mse:.8f}")

# ============================================================
# RUN EXPERIMENT
# ============================================================

f_egarch, y = egarch_t_forecast(returns, train_size)
f_ewt_egarch, _ = ewt_egarch_forecast(returns, train_size, EWT_ALPHA, EWT_K)
f_ewt_arima_egarch, _ = ewt_arima_egarch_forecast(
    returns,
    train_size,
    ARIMA_ORDER,
    EWT_ALPHA,
    EWT_K
)

print("\n===== IMPROVED VOLATILITY BENCHMARK =====\n")
evaluate("EGARCH(1,1)-t", f_egarch, y)
evaluate("EWT(low) → EGARCH-t", f_ewt_egarch, y)
evaluate("EWT(low) → ARIMA → EGARCH-t", f_ewt_arima_egarch, y)
