import numpy as np
import pandas as pd
import logging
import warnings
import traceback

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
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger()

# ======================================================
# Smooth Transition
# ======================================================
def smooth_beta(x):
    x = np.clip(x, 0.0, 1.0)
    return x**4 * (35 - 84*x + 70*x**2 - 20*x**3)

# ======================================================
# EWT Boundary Detection (TRAIN ONLY)
# Returns boundaries as normalized frequency in cycles/sample (0..0.5)
# ======================================================
def detect_boundaries(signal, max_modes):
    N = len(signal)
    if N < 8:
        raise ValueError("Signal too short for boundary detection")

    spectrum = np.abs(fft(signal))[: N // 2]
    spectrum[0] = 0.0  # remove DC

    # find peaks with minimum distance so we don't pick adjacent bins
    peaks, _ = find_peaks(spectrum, distance=max(1, (N//2)//max(1, max_modes)))
    if len(peaks) < 2:
        raise RuntimeError("Not enough spectral peaks for EWT boundaries")

    # select top max_modes peaks by magnitude, then sort by frequency
    peaks = peaks[np.argsort(spectrum[peaks])[::-1]]
    peaks = np.sort(peaks[:max_modes])

    # convert peak indices -> normalized frequencies and take midpoints as boundaries
    freqs = peaks / float(N)
    boundaries = 0.5 * (freqs[:-1] + freqs[1:])
    log.info(f"Detected {len(peaks)} peaks, producing {len(boundaries)} boundaries")
    return boundaries

# ======================================================
# EWT Decomposition (adds missing high-frequency band)
# boundaries: array of normalized frequencies (0..0.5)
# gamma: relative transition width (e.g. 0.2)
# ======================================================
def ewt_decompose(signal, boundaries, gamma):
    N = len(signal)
    if N < 4:
        raise ValueError("Signal too short for EWT decomposition")

    # absolute frequencies for each FFT bin, in cycles/sample
    freqs = np.abs(fftfreq(N))
    fft_sig = fft(signal)

    filters = []

    # ensure boundaries sorted
    b = np.sort(np.asarray(boundaries))
    # clamp boundaries to (0, 0.5)
    b = np.clip(b, 1e-12, 0.5 - 1e-12)

    # low-pass (below first boundary)
    w1 = b[0]
    phi = np.zeros(N)
    low = freqs <= (1 - gamma) * w1
    trans = ((1 - gamma) * w1 < freqs) & (freqs <= (1 + gamma) * w1)
    phi[low] = 1.0
    phi[trans] = np.cos(
        (np.pi / 2) * smooth_beta((freqs[trans] - (1 - gamma) * w1) / (2 * gamma * w1))
    )
    filters.append(phi)

    # interior bandpass filters
    for w0, w1 in zip(b[:-1], b[1:]):
        psi = np.zeros(N)

        # central flat band
        band = ((1 + gamma) * w0 <= freqs) & (freqs <= (1 - gamma) * w1)
        psi[band] = 1.0

        # upper transition (towards w1)
        up = ((1 - gamma) * w1 < freqs) & (freqs <= (1 + gamma) * w1)
        psi[up] = np.cos(
            (np.pi / 2) * smooth_beta((freqs[up] - (1 - gamma) * w1) / (2 * gamma * w1))
        )

        # lower transition (away from w0)
        down = ((1 - gamma) * w0 <= freqs) & (freqs <= (1 + gamma) * w0)
        psi[down] = np.sin(
            (np.pi / 2) * smooth_beta((freqs[down] - (1 - gamma) * w0) / (2 * gamma * w0))
        )

        filters.append(psi)

    # high-pass (above last boundary) — THIS WAS MISSING
    wlast = b[-1]
    psi_last = np.zeros(N)
    high = freqs >= (1 + gamma) * wlast
    trans_last = ((1 - gamma) * wlast <= freqs) & (freqs <= (1 + gamma) * wlast)
    psi_last[high] = 1.0
    # smooth rise from 0 to 1 in the transition zone; use sin to complement lowpass
    psi_last[trans_last] = np.sin(
        (np.pi / 2) * smooth_beta((freqs[trans_last] - (1 - gamma) * wlast) / (2 * gamma * wlast))
    )
    filters.append(psi_last)

    # apply filters and inverse FFT to get components
    comps = []
    for idx, f in enumerate(filters):
        comp = np.real(ifft(fft_sig * f))
        comps.append(comp)

    log.info(f"EWT produced {len(comps)} components (filters)")
    return comps

# ======================================================
# Realized Variance Proxy
# ======================================================
def realized_variance(returns, window=5):
    # returns: 1D array-like of returns
    s = pd.Series(returns).astype(float)
    rv = s.rolling(window).apply(lambda x: np.sum(x**2), raw=True)
    return rv.values  # numpy array with NaNs initially

# ======================================================
# Benchmarks (defensive)
# ======================================================
def ar1_rv_forecast(rv):
    try:
        last = float(np.asarray(rv)[-1])
        return max(last, 1e-12)
    except Exception:
        log.warning("AR1 fallback: bad input, returning tiny value")
        return 1e-12

def arima_rv_forecast(rv):
    try:
        model = ARIMA(rv, order=(1,0,1))
        res = model.fit()
        f = float(res.forecast(steps=1)[0])
        return max(f, 1e-12)
    except Exception as e:
        log.warning(f"ARIMA forecast failed: {e}; falling back to AR1")
        return ar1_rv_forecast(rv)

def garch_forecast(ret):
    try:
        model = arch_model(ret, mean="Zero", vol="GARCH", p=1, q=1, dist="t", rescale=False)
        res = model.fit(disp="off")
        # variance forecast for last observation, horizon=1
        vf = res.forecast(horizon=1).variance
        # vf may be a DataFrame-like; handle safely
        if hasattr(vf, "values"):
            val = vf.values[-1, 0]
        else:
            val = np.asarray(vf)[-1, 0]
        return max(float(val), 1e-12)
    except Exception as e:
        log.warning(f"GARCH fit failed: {e}; returning sample variance fallback")
        return max(float(np.nanvar(ret)), 1e-12)

# ======================================================
# EWT model (fit ARIMA to each component, robust)
# ======================================================
def ewt_rv_forecast(rv, boundaries, gamma):
    try:
        comps = ewt_decompose(rv, boundaries, gamma)
    except Exception as e:
        log.warning(f"EWT decomposition failed: {e}; returning AR1")
        return ar1_rv_forecast(rv)

    rv_hat = 0.0
    for c in comps:
        # skip near-constant components
        if np.nanstd(c) < 1e-8:
            continue
        try:
            # fit ARIMA(1,0,1) on component
            arima_res = ARIMA(c, order=(1,0,1)).fit()
            f = float(arima_res.forecast(steps=1)[0])
            rv_hat += max(f, 0.0)
        except Exception as e:
            log.debug(f"Component ARIMA failed: {e}; skipping component")
            continue

    return max(rv_hat, 1e-12)

# ======================================================
# Hybrid model — more coherent scaling using mean recent realized variance
# ======================================================
def hybrid_forecast(rv_hist, ret_hist, boundaries, gamma):
    # rv_hist : recent realized variance series (same units as rv forecasts)
    # ret_hist: recent returns (to feed GARCH)
    rv_struct = ewt_rv_forecast(rv_hist, boundaries, gamma)
    garch_var = garch_forecast(ret_hist)

    # Use mean realized variance of the same history to map scales (more consistent)
    rv_mean = np.nanmean(rv_hist)
    if rv_mean <= 0 or np.isnan(rv_mean):
        log.warning("rv_mean nonpositive or NaN; falling back to unity scale")
        scale = 1.0
    else:
        scale = garch_var / rv_mean

    return max(rv_struct * scale, 1e-12)

# ======================================================
# MAIN (unchanged control flow, but safer)
# ======================================================
if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv")
    df["ret"] = np.log(df["close"]).diff()
    df.dropna(inplace=True)

    rv = realized_variance(df["ret"].values, window=5)
    # drop initial NaNs from rolling
    df = df.iloc[5:].reset_index(drop=True)
    rv = rv[5:]
    returns = df["ret"].values

    # Train / Test split
    split = int(len(rv) * 0.7)
    train_rv = rv[:split]
    test_rv = rv[split:]
    train_ret = returns[:split]
    test_ret = returns[split:]

    log.info(f"Train size: {len(train_rv)} | Test size: {len(test_rv)}")

    # Learn EWT boundaries on TRAIN (use variance training series without NaNs)
    max_modes = 5
    gamma = 0.2

    # drop NaNs before detecting boundaries
    train_rv_clean = train_rv[np.isfinite(train_rv)]
    if len(train_rv_clean) < 32:
        raise RuntimeError("Not enough valid train rv to detect EWT boundaries")

    boundaries = detect_boundaries(train_rv_clean, max_modes)
    log.info("EWT boundaries fixed")

    # Rolling OOS Forecast
    window = 800
    if window >= len(test_rv):
        raise ValueError("Window too large for test set")

    y_true = []
    pred_ar1 = []
    pred_arima = []
    pred_ewt = []
    pred_hybrid = []

    for i in range(window, len(test_rv)):
        try:
            rv_hist = test_rv[i - window:i]
            ret_hist = test_ret[i - window:i]
            true_val = float(test_rv[i])

            y_true.append(true_val)

            pred_ar1.append(ar1_rv_forecast(rv_hist))
            pred_arima.append(arima_rv_forecast(rv_hist))
            pred_ewt.append(ewt_rv_forecast(rv_hist, boundaries, gamma))
            pred_hybrid.append(hybrid_forecast(rv_hist, ret_hist, boundaries, gamma))

        except Exception as e:
            log.error(f"OOS iteration failed at i={i}: {e}\n{traceback.format_exc()}")
            continue

        if (i - window) % 100 == 0:
            log.info(f"OOS {i - window}/{len(test_rv) - window}")

    # Convert variance → volatility (safe clipping)
    y_true = np.sqrt(np.clip(np.array(y_true, dtype=float), 0.0, None))
    def to_vol(arr):
        return np.sqrt(np.clip(np.array(arr, dtype=float), 0.0, None))

    pred_ar1 = to_vol(pred_ar1)
    pred_arima = to_vol(pred_arima)
    pred_ewt = to_vol(pred_ewt)
    pred_hybrid = to_vol(pred_hybrid)

    # Metrics
    log.info("====== OOS RESULTS ======")
    names = ["AR1", "ARIMA", "EWT", "HYBRID"]
    preds = [pred_ar1, pred_arima, pred_ewt, pred_hybrid]

    for n, p in zip(names, preds):
        if len(p) == 0:
            log.info(f"{n} | no predictions produced")
            continue
        mse = mean_squared_error(y_true, p)
        mae = mean_absolute_error(y_true, p)
        log.info(f"{n} | MSE={mse:.6f} | MAE={mae:.6f}")
