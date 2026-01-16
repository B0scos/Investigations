import numpy as np
import pandas as pd
import logging
import warnings

from numpy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks

from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

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
# Smooth transition
# ======================================================

def smooth_beta(x):
    x = np.clip(x, 0, 1)
    return x**4 * (35 - 84*x + 70*x**2 - 20*x**3)

# ======================================================
# EWT Boundary Detection (TRAIN ONLY)
# ======================================================

def detect_boundaries(signal, max_modes):

    N = len(signal)

    spectrum = np.abs(fft(signal))[:N//2]
    spectrum[0] = 0

    peaks, _ = find_peaks(
        spectrum,
        distance=max(1, (N//2)//max_modes)
    )

    if len(peaks) < 2:
        raise RuntimeError("Not enough spectral peaks")

    peaks = peaks[np.argsort(spectrum[peaks])[::-1]]
    peaks = np.sort(peaks[:max_modes])

    return 0.5 * (peaks[:-1] + peaks[1:]) / N

# ======================================================
# EWT Decomposition
# ======================================================

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
        np.pi/2 * smooth_beta(
            (freqs[trans] - (1-gamma)*w1) / (2*gamma*w1)
        )
    )

    filters.append(phi)

    for w0, w1 in zip(boundaries[:-1], boundaries[1:]):

        psi = np.zeros(N)

        band = ((1+gamma)*w0 <= freqs) & (freqs <= (1-gamma)*w1)
        up = ((1-gamma)*w1 <= freqs) & (freqs <= (1+gamma)*w1)
        down = ((1-gamma)*w0 <= freqs) & (freqs <= (1+gamma)*w0)

        psi[band] = 1

        psi[up] = np.cos(
            np.pi/2 * smooth_beta(
                (freqs[up] - (1-gamma)*w1) / (2*gamma*w1)
            )
        )

        psi[down] = np.sin(
            np.pi/2 * smooth_beta(
                (freqs[down] - (1-gamma)*w0) / (2*gamma*w0)
            )
        )

        filters.append(psi)

    comps = [np.real(ifft(fft_sig * f)) for f in filters]

    return comps

# ======================================================
# Realized Variance Proxy
# ======================================================

def realized_variance(returns, window=5):
    return pd.Series(returns).rolling(window).apply(
        lambda x: np.sum(x**2),
        raw=True
    ).values

# ======================================================
# Plain GARCH (RETURNS ONLY)
# ======================================================

def garch_forecast(returns):

    model = arch_model(
        returns,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=False
    )

    res = model.fit(disp="off")

    return res.forecast(horizon=1).variance.values[-1, 0]

# ======================================================
# EWT + ARIMA on RV components
# ======================================================

def ewt_rv_forecast(rv_series, boundaries, gamma):

    comps = ewt_decompose(rv_series, boundaries, gamma)

    rv_hat = 0.0

    for c in comps:

        if np.std(c) < 1e-8:
            continue

        arima = ARIMA(c, order=(1,0,1)).fit()

        rv_hat += arima.forecast()[0]

    return max(rv_hat, 1e-12)

# ======================================================
# Hybrid: EWT structure + GARCH shocks
# ======================================================

def hybrid_forecast(rv_series, ret_series, boundaries, gamma):

    rv_hat = ewt_rv_forecast(rv_series, boundaries, gamma)

    garch_var = garch_forecast(ret_series)

    # scale RV structure with return shock dynamics
    scale = garch_var / np.mean(ret_series**2)

    return rv_hat * scale

# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    df = pd.read_csv(
        r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv"
    )

    df["ret"] = np.log(df["close"]).diff()
    df.dropna(inplace=True)

    rv = realized_variance(df["ret"].values, window=5)

    df = df.iloc[5:]
    rv = rv[5:]

    returns = df["ret"].values

    # -----------------------------
    # TRAIN / TEST SPLIT
    # -----------------------------

    split = int(len(rv) * 0.7)

    train_rv = rv[:split]
    test_rv = rv[split:]

    train_ret = returns[:split]
    test_ret = returns[split:]

    log.info(f"Train size: {len(train_rv)}")
    log.info(f"Test size : {len(test_rv)}")

    # -----------------------------
    # Learn EWT on TRAIN only
    # -----------------------------

    max_modes = 5
    gamma = 0.2

    boundaries = detect_boundaries(train_rv, max_modes)

    log.info(f"EWT boundaries fixed")

    # -----------------------------
    # Out-of-sample rolling test
    # -----------------------------

    window = 800

    y_true = []
    pred_plain = []
    pred_ewt = []
    pred_hybrid = []

    for i in range(window, len(test_rv)):

        rv_hist = test_rv[i-window:i]
        ret_hist = test_ret[i-window:i]

        try:

            v_plain = garch_forecast(ret_hist)

            v_ewt = ewt_rv_forecast(rv_hist, boundaries, gamma)

            v_hybrid = hybrid_forecast(
                rv_hist,
                ret_hist,
                boundaries,
                gamma
            )

            y_true.append(test_rv[i])

            pred_plain.append(v_plain)
            pred_ewt.append(v_ewt)
            pred_hybrid.append(v_hybrid)

        except:
            continue

        if i % 100 == 0:
            log.info(f"OOS {i-window}/{len(test_rv)-window}")

    y_true = np.sqrt(np.array(y_true))
    pred_plain = np.sqrt(np.array(pred_plain))
    pred_ewt = np.sqrt(np.array(pred_ewt))
    pred_hybrid = np.sqrt(np.array(pred_hybrid))

    # -----------------------------
    # Metrics
    # -----------------------------

    log.info("====== OOS RESULTS ======")

    for name, pred in zip(
        ["PLAIN", "EWT", "HYBRID"],
        [pred_plain, pred_ewt, pred_hybrid]
    ):

        mse = mean_squared_error(y_true, pred)
        mae = mean_absolute_error(y_true, pred)

        log.info(f"{name} | MSE={mse:.6f} | MAE={mae:.6f}")
