# ============================================================
# EWT + ARIMA + GARCH Volatility Benchmark (BTC log-returns)
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
# DATA LOADING
# ============================================================

CSV_PATH = r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv"     # must contain price column
PRICE_COL = "close"
DATE_COL = "timestamp"

TRAIN_RATIO = 0.8
ALPHA_EWT = 0.05
ARIMA_ORDER = (1, 0, 1)


df = pd.read_csv(CSV_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL)

prices = df[PRICE_COL].values
returns = np.diff(np.log(prices))
returns = returns[~np.isnan(returns)]


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
        mode = np.real(ifft(Xf * mask))
        modes.append(mode)

    return modes


# ============================================================
# GARCH FORECASTING
# ============================================================

def garch_forecast(returns, train_size):
    forecasts = []
    actual = []

    for t in range(train_size, len(returns) - 1):
        train = returns[:t]

        model = arch_model(
            train,
            vol="GARCH",
            p=1,
            q=1,
            mean="Zero",
            rescale=False
        )
        res = model.fit(disp="off")

        var = res.forecast(horizon=1).variance.iloc[-1, 0]
        forecasts.append(var)
        actual.append(returns[t + 1] ** 2)

    return np.array(forecasts), np.array(actual)


# ============================================================
# EWT → GARCH
# ============================================================

def ewt_garch_forecast(returns, train_size, alpha):
    modes = ewt_decompose(returns, alpha)
    reconstructed = np.sum(modes, axis=0)
    return garch_forecast(reconstructed, train_size)


# ============================================================
# EWT → ARIMA
# ============================================================

def ewt_arima_forecast(returns, train_size, order, alpha):
    modes = ewt_decompose(returns, alpha)
    signal = np.sum(modes, axis=0)

    forecasts = []
    actual = []

    for t in range(train_size, len(signal) - 1):
        model = ARIMA(signal[:t], order=order)
        res = model.fit()

        pred = res.forecast()[0]
        forecasts.append(pred ** 2)
        actual.append(signal[t + 1] ** 2)

    return np.array(forecasts), np.array(actual)


# ============================================================
# EWT → ARIMA → GARCH
# ============================================================

def ewt_arima_garch_forecast(returns, train_size, arima_order, alpha):
    modes = ewt_decompose(returns, alpha)
    signal = np.sum(modes, axis=0)

    forecasts = []
    actual = []

    for t in range(train_size, len(signal) - 1):
        arima = ARIMA(signal[:t], order=arima_order).fit()
        resid = arima.resid

        garch = arch_model(
            resid,
            vol="GARCH",
            p=1,
            q=1,
            mean="Zero",
            rescale=False
        )
        garch_res = garch.fit(disp="off")

        var = garch_res.forecast(horizon=1).variance.iloc[-1, 0]
        forecasts.append(var)
        actual.append(signal[t + 1] ** 2)

    return np.array(forecasts), np.array(actual)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(name, yhat, y):
    mse = mean_squared_error(y, yhat)
    print(f"{name:<30} MSE = {mse:.8f}")


# ============================================================
# RUN EXPERIMENT
# ============================================================

train_size = int(len(returns) * TRAIN_RATIO)

f_garch, y = garch_forecast(returns, train_size)
f_ewt_garch, _ = ewt_garch_forecast(returns, train_size, ALPHA_EWT)
f_ewt_arima, _ = ewt_arima_forecast(returns, train_size, ARIMA_ORDER, ALPHA_EWT)
f_ewt_arima_garch, _ = ewt_arima_garch_forecast(
    returns, train_size, ARIMA_ORDER, ALPHA_EWT
)

print("\n===== VOLATILITY FORECAST BENCHMARK =====\n")
evaluate("GARCH(1,1)", f_garch, y)
evaluate("EWT → GARCH", f_ewt_garch, y)
evaluate("EWT → ARIMA", f_ewt_arima, y)
evaluate("EWT → ARIMA → GARCH", f_ewt_arima_garch, y)
