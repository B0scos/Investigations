import numpy as np
import pandas as pd
import logging
import time
from numpy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import norm
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ==================================================
# Logging config
# ==================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)


# ==================================================
# Smooth transition function β(x)
# ==================================================
def beta(x):
    x = np.clip(x, 0.0, 1.0)
    return x**4 * (35 - 84*x + 70*x**2 - 20*x**3)


# ==================================================
# EWT boundary detection
# ==================================================
def detect_boundaries(signal, max_modes=6):

    N = len(signal)

    spectrum = np.abs(fft(signal))[: N // 2]
    spectrum[0] = 0.0

    peaks, _ = find_peaks(
        spectrum,
        distance=max(1, (N // 2) // max_modes)
    )

    if len(peaks) < 2:
        raise RuntimeError("Not enough spectral peaks")

    peaks = peaks[np.argsort(spectrum[peaks])[::-1]]
    peaks = np.sort(peaks[:max_modes])

    boundaries = 0.5 * (peaks[:-1] + peaks[1:])

    return boundaries / N


# ==================================================
# Empirical Wavelet Transform
# ==================================================
def ewt_decompose(signal, max_modes=6, gamma=0.25):

    N = len(signal)

    freqs = fftfreq(N)
    abs_freqs = np.abs(freqs)

    fft_signal = fft(signal)

    boundaries = detect_boundaries(signal, max_modes)

    filters = []

    # ---- Scaling function φ ----
    w1 = boundaries[0]

    phi = np.zeros(N)

    mask1 = abs_freqs <= (1 - gamma) * w1
    mask2 = ((1 - gamma) * w1 < abs_freqs) & (abs_freqs <= (1 + gamma) * w1)

    phi[mask1] = 1.0
    phi[mask2] = np.cos(
        np.pi / 2 * beta(
            (abs_freqs[mask2] - (1 - gamma) * w1) / (2 * gamma * w1)
        )
    )

    filters.append(phi)

    # ---- Wavelets ψ_n ----
    for wn, wnp1 in zip(boundaries[:-1], boundaries[1:]):

        psi = np.zeros(N)

        band = ((1 + gamma) * wn <= abs_freqs) & (abs_freqs <= (1 - gamma) * wnp1)
        up = ((1 - gamma) * wnp1 <= abs_freqs) & (abs_freqs <= (1 + gamma) * wnp1)
        down = ((1 - gamma) * wn <= abs_freqs) & (abs_freqs <= (1 + gamma) * wn)

        psi[band] = 1.0

        psi[up] = np.cos(
            np.pi / 2 * beta(
                (abs_freqs[up] - (1 - gamma) * wnp1) / (2 * gamma * wnp1)
            )
        )

        psi[down] = np.sin(
            np.pi / 2 * beta(
                (abs_freqs[down] - (1 - gamma) * wn) / (2 * gamma * wn)
            )
        )

        filters.append(psi)

    components = [
        np.real(ifft(fft_signal * filt))
        for filt in filters
    ]

    return components


# ==================================================
# GARCH fitting (safe wrapper)
# ==================================================
def fit_garch(series):

    series = np.asarray(series)
    series = series[~np.isnan(series)]

    if len(series) < 50:
        raise RuntimeError("Series too short")

    model = arch_model(
        series,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=True
    )

    res = model.fit(disp="off")

    return res


# ==================================================
# Rolling OOS evaluation (optimized + logged)
# ==================================================
def rolling_oos_eval(
    returns,
    realized_vol,
    window,
    max_modes=6,
    gamma=0.25
):

    n_total = len(returns)
    n_steps = n_total - window

    logger.info(f"OOS steps: {n_steps}")

    garch_preds = np.full(n_steps, np.nan)
    ewt_preds = np.full(n_steps, np.nan)
    rv_true = np.full(n_steps, np.nan)

    t0 = time.time()

    for i, t in enumerate(range(window, n_total)):

        train = returns[t - window: t]

        # ---------- Plain GARCH ----------
        try:
            res_g = fit_garch(train)
            garch_preds[i] = np.sqrt(
                res_g.forecast(horizon=1).variance.values[-1, 0]
            )
        except Exception as e:
            logger.warning(f"GARCH failed at t={t}: {e}")

        # ---------- EWT + GARCH ----------
        try:
            components = ewt_decompose(
                train,
                max_modes=max_modes,
                gamma=gamma
            )

            vols = []

            for comp in components:

                if np.std(comp) < 1e-10:
                    continue

                res_c = fit_garch(comp)

                v = np.sqrt(
                    res_c.forecast(horizon=1).variance.values[-1, 0]
                )

                vols.append(v)

            if vols:
                ewt_preds[i] = np.sqrt(np.sum(np.square(vols)))

        except Exception as e:
            logger.warning(f"EWT failed at t={t}: {e}")

        rv_true[i] = realized_vol[t]

        # ---------- Progress ----------
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - t0
            speed = i / elapsed
            logger.info(f"{i}/{n_steps} | {speed:.2f} steps/sec")

    mask = ~np.isnan(garch_preds) & ~np.isnan(ewt_preds)

    if not np.any(mask):
        raise RuntimeError("No valid forecasts")

    logger.info(f"Valid forecasts: {np.sum(mask)}")

    return {
        "rv": rv_true[mask],
        "garch": garch_preds[mask],
        "ewt": ewt_preds[mask],
    }


# ==================================================
# Diebold–Mariano test
# ==================================================
def diebold_mariano(y, f1, f2, loss="mse"):

    y, f1, f2 = map(np.asarray, (y, f1, f2))

    if loss == "mse":
        d = (y - f1)**2 - (y - f2)**2
    elif loss == "mae":
        d = np.abs(y - f1) - np.abs(y - f2)
    else:
        raise ValueError("Invalid loss")

    T = len(d)

    d_bar = np.mean(d)
    var_d = np.var(d, ddof=0)

    DM = d_bar / np.sqrt(var_d / T)
    p_value = 2 * (1 - norm.cdf(abs(DM)))

    return DM, p_value


# ==================================================
# Main
# ==================================================
if __name__ == "__main__":

    start_all = time.time()

    df = pd.read_csv(
        r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv"
    )

    df = df[df["timestamp"] >= "2019-01-01"]

    df["ret"] = np.log(df["close"]).diff()
    df["rv"] = np.abs(df["ret"])

    df.dropna(inplace=True)

    window = 1000

    logger.info(f"Window size: {window}")
    logger.info(f"Total observations: {len(df)}")

    results = rolling_oos_eval(
        returns=df["ret"].values,
        realized_vol=df["rv"].values,
        window=window,
        max_modes=6,
        gamma=0.3
    )

    # ---------- Metrics ----------
    logger.info("===== OOS METRICS =====")

    garch_mse = mean_squared_error(results["rv"], results["garch"])
    ewt_mse = mean_squared_error(results["rv"], results["ewt"])

    garch_mae = mean_absolute_error(results["rv"], results["garch"])
    ewt_mae = mean_absolute_error(results["rv"], results["ewt"])

    logger.info(f"GARCH MSE     : {garch_mse:.6f}")
    logger.info(f"EWT-GARCH MSE : {ewt_mse:.6f}")

    logger.info(f"GARCH MAE     : {garch_mae:.6f}")
    logger.info(f"EWT-GARCH MAE : {ewt_mae:.6f}")

    # ---------- DM Test ----------
    DM, p = diebold_mariano(
        results["rv"],
        results["garch"],
        results["ewt"],
        loss="mse"
    )

    DM_mae, p_mae = diebold_mariano(
    results["rv"],
    results["garch"],
    results["ewt"],
    loss="mae"
)


    logger.info("===== DIEBOLD–MARIANO mse =====")
    logger.info(f"DM statistic : {DM:.4f}")
    logger.info(f"p-value      : {p:.6f}")

    logger.info("===== DIEBOLD–MARIANO mae =====")
    logger.info(f"DM statistic : {DM_mae:.4f}")
    logger.info(f"p-value      : {p_mae:.6f}")

    logger.info(f"TOTAL RUNTIME: {(time.time() - start_all)/60:.2f} minutes")
    logger.info(f"p-value      : {p:.6f}")

    logger.info(f"TOTAL RUNTIME: {(time.time() - start_all)/60:.2f} minutes")

    logger.info(f"TOTAL RUNTIME: {(time.time() - start_all)/60:.2f} minutes")

    
