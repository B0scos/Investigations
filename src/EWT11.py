import numpy as np
import pandas as pd
import logging
from numpy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import norm
import warnings

# ======================================================
# Logging
# ======================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)

log = logging.getLogger()


# ======================================================
# Utils
# ======================================================

def smooth_beta(x):
    x = np.clip(x, 0, 1)
    return x**4 * (35 - 84*x + 70*x**2 - 20*x**3)


# ======================================================
# EWT
# ======================================================

def detect_boundaries(signal, max_modes):

    N = len(signal)
    spectrum = np.abs(fft(signal))[:N//2]
    spectrum[0] = 0

    peaks, _ = find_peaks(
        spectrum,
        distance=max(1, (N // 2) // max_modes)
    )

    if len(peaks) < 2:
        raise RuntimeError("Insufficient spectral peaks")

    peaks = peaks[np.argsort(spectrum[peaks])[::-1]]
    peaks = np.sort(peaks[:max_modes])

    return 0.5 * (peaks[:-1] + peaks[1:]) / N


def ewt_decompose(signal, boundaries, gamma):

    N = len(signal)
    freqs = np.abs(fftfreq(N))
    fft_sig = fft(signal)

    filters = []

    w1 = boundaries[0]

    phi = np.zeros(N)

    low = freqs <= (1 - gamma) * w1
    trans = ((1 - gamma) * w1 < freqs) & (freqs <= (1 + gamma) * w1)

    phi[low] = 1
    phi[trans] = np.cos(
        np.pi / 2 * smooth_beta(
            (freqs[trans] - (1 - gamma) * w1) / (2 * gamma * w1)
        )
    )

    filters.append(phi)

    for w0, w1 in zip(boundaries[:-1], boundaries[1:]):

        psi = np.zeros(N)

        band = ((1 + gamma) * w0 <= freqs) & (freqs <= (1 - gamma) * w1)
        up = ((1 - gamma) * w1 <= freqs) & (freqs <= (1 + gamma) * w1)
        down = ((1 - gamma) * w0 <= freqs) & (freqs <= (1 + gamma) * w0)

        psi[band] = 1

        psi[up] = np.cos(
            np.pi / 2 * smooth_beta(
                (freqs[up] - (1 - gamma) * w1) / (2 * gamma * w1)
            )
        )

        psi[down] = np.sin(
            np.pi / 2 * smooth_beta(
                (freqs[down] - (1 - gamma) * w0) / (2 * gamma * w0)
            )
        )

        filters.append(psi)

    components = [np.real(ifft(fft_sig * f)) for f in filters]

    return components


# ======================================================
# Models
# ======================================================

def fit_garch(x):

    x = x[~np.isnan(x)]

    if len(x) < 100:
        raise RuntimeError("Series too short")

    model = arch_model(
        x,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=False
    )

    return model.fit(disp="off")


def garch_forecast_variance(x):

    res = fit_garch(x)
    return res.forecast(horizon=1).variance.values[-1, 0]


def svm_forecast_variance(series, lags=3):

    v = series**2

    X, y = [], []

    for i in range(lags, len(v) - 1):
        X.append(v[i-lags:i])
        y.append(v[i])

    X, y = np.array(X), np.array(y)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = SVR(kernel="rbf", C=10, epsilon=1e-5)
    model.fit(Xs, y)

    last = scaler.transform(v[-lags:].reshape(1, -1))
    pred = model.predict(last)[0]

    return max(pred, 0)

def hybrid_ewt_arima_garch(returns, boundaries, gamma):

    comps = ewt_decompose(returns, boundaries, gamma)

    # ===============================
    # LOW FREQUENCY: ARIMA + GARCH
    # ===============================

    low = comps[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        arima = ARIMA(
            low,
            order=(1, 0, 1),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit()

    mu_forecast = arima.forecast()[0]

    resid = low - arima.fittedvalues

    v_low = garch_forecast_variance(resid)

    # ===============================
    # HIGH FREQUENCY: GARCH ONLY
    # ===============================

    v_high = 0.0

    for c in comps[1:]:

        if np.std(c) < 1e-8:
            continue

        v_high += garch_forecast_variance(c)

    # ===============================
    # AGGREGATION
    # ===============================

    total_variance = v_low + v_high
    total_volatility = np.sqrt(total_variance)

    return mu_forecast, total_volatility


# ======================================================
# Realized Variance Proxy
# ======================================================

def realized_variance(returns, window=5):
    return pd.Series(returns).rolling(window).apply(lambda x: np.sum(x**2)).values


# ======================================================
# Rolling Evaluation
# ======================================================

def rolling_backtest(
    returns,
    rv,
    window,
    max_modes=6,
    gamma=0.2
):

    n = len(returns)

    preds = {
        "rv": [],
        "garch_plain": [],
        "garch_ewt": [],
        "svm_ewt": [],
        "hybrid_arima_garch": []
    }

    for t in range(window, n):

        train = returns[:t]

        try:
            # ============================
            # Plain GARCH
            # ============================

            v_plain = garch_forecast_variance(train)

            # ============================
            # EWT Decomposition
            # ============================

            boundaries = detect_boundaries(train, max_modes)
            comps = ewt_decompose(train, boundaries, gamma)

            # ============================
            # EWT + GARCH
            # ============================

            v_ewt = 0.0

            for c in comps:
                if np.std(c) < 1e-8:
                    continue
                v_ewt += garch_forecast_variance(c)

            # ============================
            # EWT + SVM
            # ============================

            v_svm = 0.0

            for c in comps:
                if np.std(c) < 1e-8:
                    continue
                v_svm += svm_forecast_variance(c)

            # ============================
            # HYBRID: EWT + ARIMA + GARCH
            # ============================

            low = comps[0]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                arima = ARIMA(
                    low,
                    order=(1, 0, 1),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit()

            resid = low - arima.fittedvalues

            v_low = garch_forecast_variance(resid)

            v_high = 0.0

            for c in comps[1:]:
                if np.std(c) < 1e-8:
                    continue
                v_high += garch_forecast_variance(c)

            v_hybrid = v_low + v_high

            # ============================
            # Store (convert variance → vol)
            # ============================

            preds["rv"].append(np.sqrt(rv[t]))
            preds["garch_plain"].append(np.sqrt(v_plain))
            preds["garch_ewt"].append(np.sqrt(v_ewt))
            preds["svm_ewt"].append(np.sqrt(v_svm))
            preds["hybrid_arima_garch"].append(np.sqrt(v_hybrid))

        except Exception:
            continue

        if (t - window) % 100 == 0:
            log.info(f"{t-window}/{n-window}")

    return {k: np.array(v) for k, v in preds.items()}



# ======================================================
# Diebold-Mariano (HAC)
# ======================================================

def diebold_mariano(y, f1, f2, h=1):

    d = (y - f1)**2 - (y - f2)**2

    T = len(d)

    gamma0 = np.var(d)

    cov = 0
    for lag in range(1, h):
        weight = 1 - lag / h
        cov += 2 * weight * np.cov(d[:-lag], d[lag:])[0, 1]

    var_d = (gamma0 + cov) / T

    DM = np.mean(d) / np.sqrt(var_d)
    p = 2 * (1 - norm.cdf(abs(DM)))

    return DM, p


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    df = pd.read_csv(r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv")
    

    df["ret"] = np.log(df["close"]).diff()
    df = df[df["timestamp"] >= "2023-01-01"]
    df.dropna(inplace=True)

    rv = realized_variance(df["ret"].values, window=3)

    window = 1000

    results = rolling_backtest(
        returns=df["ret"].values,
        rv=rv,
        window=window,
        max_modes=5,
        gamma=0.2
    )

    log.info("====== METRICS ======")

    for k in results.keys():
        if k == "rv":
            continue

        mse = mean_squared_error(results["rv"], results[k])
        mae = mean_absolute_error(results["rv"], results[k])

        log.info(f"{k.upper()} | MSE={mse:.6f} | MAE={mae:.6f}")

    DM1, p1 = diebold_mariano(
        results["rv"],
        results["garch_plain"],
        results["garch_ewt"]
    )

    DM2, p2 = diebold_mariano(
        results["rv"],
        results["garch_plain"],
        results["svm_ewt"]
    )

    DM3, p3 = diebold_mariano(
        results["rv"],
        results["garch_plain"],
        results["hybrid_arima_garch"]
    )

    log.info("====== DM TEST ======")
    log.info(f"GARCH vs EWT-GARCH : DM={DM1:.3f} p={p1:.6f}")
    log.info(f"GARCH vs EWT-SVM   : DM={DM2:.3f} p={p2:.6f}")
    log.info(f"GARCH vs hybrid_arima_garch   : DM={DM2:.3f} p={p2:.6f}")
