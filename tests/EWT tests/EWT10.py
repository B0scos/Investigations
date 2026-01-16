import numpy as np
import pandas as pd
import logging
import time
from numpy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import norm
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
import warnings


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
# Smooth transition function
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
# EWT decomposition
# ==================================================
def ewt_decompose(signal, max_modes=6, gamma=0.25):

    N = len(signal)

    freqs = fftfreq(N)
    abs_freqs = np.abs(freqs)

    fft_signal = fft(signal)

    boundaries = detect_boundaries(signal, max_modes)

    filters = []

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
# GARCH
# ==================================================
def fit_garch(series):

    series = np.asarray(series)
    series = series[~np.isnan(series)]

    if len(series) < 50:
        raise RuntimeError("Series too short")

    model = arch_model(
        series,
        mean="constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=False
    )

    return model.fit(disp="off")


# ==================================================
# Rolling OOS
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

    garch_plain = np.full(n_steps, np.nan)
    garch_ewt = np.full(n_steps, np.nan)
    arima_low_garch_high = np.full(n_steps, np.nan)
    svm_ewt = np.full(n_steps, np.nan)
    rv_true = np.full(n_steps, np.nan)

    t0 = time.time()

    for i, t in enumerate(range(window, n_total)):

        train = returns[t - window: t]

        # ---------- Plain GARCH ----------
        try:
            res_g = fit_garch(train)
            garch_plain[i] = np.sqrt(
                res_g.forecast(horizon=1).variance.values[-1, 0]
            )
        except Exception as e:
            logger.warning(f"Plain GARCH failed at t={t}: {e}")

        # ---------- EWT pipeline ----------
        try:
            components = ewt_decompose(train, max_modes=max_modes, gamma=gamma)

            # ---- EWT-GARCH ----
            vols = []
            for comp in components:
                if np.std(comp) < 1e-10:
                    continue
                res_c = fit_garch(comp)
                v = np.sqrt(res_c.forecast(horizon=1).variance.values[-1, 0])
                vols.append(v)

            if vols:
                garch_ewt[i] = np.sqrt(np.sum(np.square(vols)))

            # ---- ARIMA low + GARCH high ----
            low_freq = components[0]
            high_freqs = components[1:]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                arima = ARIMA(low_freq, order=(1, 0, 1),
                              enforce_stationarity=False,
                              enforce_invertibility=False).fit()
                low_forecast = np.abs(arima.forecast()[0])

            vols = []
            for comp in high_freqs:
                if np.std(comp) < 1e-10:
                    continue
                res_c = fit_garch(comp)
                v = np.sqrt(res_c.forecast(horizon=1).variance.values[-1, 0])
                vols.append(v)

            hf_vol = np.sqrt(np.sum(np.square(vols))) if vols else 0.0
            arima_low_garch_high[i] = low_forecast + hf_vol

            # ---- FIXED SVM VOLATILITY MODEL ----
            vols = []
            for comp in components:
                if np.std(comp) < 1e-10:
                    continue

                comp_var = comp**2

                if len(comp_var) < 5:
                    continue

                X = comp_var[:-1].reshape(-1, 1)
                y = comp_var[1:]

                model = SVR(kernel="rbf", C=1.0, epsilon=1e-6)
                model.fit(X, y)

                pred_var = model.predict(comp_var[-1].reshape(1, -1))[0]
                vols.append(max(pred_var, 0))

            if vols:
                svm_ewt[i] = np.sqrt(np.sum(vols))

        except Exception as e:
            logger.warning(f"EWT pipeline failed at t={t}: {e}")

        rv_true[i] = realized_vol[t]

        if i % 100 == 0 and i > 0:
            speed = i / (time.time() - t0)
            logger.info(f"{i}/{n_steps} | {speed:.2f} steps/sec")

    mask = (
        ~np.isnan(garch_plain) &
        ~np.isnan(garch_ewt) &
        ~np.isnan(arima_low_garch_high) &
        ~np.isnan(svm_ewt)
    )

    logger.info(f"Valid forecasts: {np.sum(mask)}")

    return {
        "rv": rv_true[mask],
        "garch_plain": garch_plain[mask],
        "garch_ewt": garch_ewt[mask],
        "arima_low_garch_high": arima_low_garch_high[mask],
        "svm_ewt": svm_ewt[mask],
    }


# ==================================================
# Diebold–Mariano
# ==================================================
def diebold_mariano(y, f1, f2, loss="mse"):

    if loss == "mse":
        d = (y - f1)**2 - (y - f2)**2
    else:
        d = np.abs(y - f1) - np.abs(y - f2)

    DM = np.mean(d) / np.sqrt(np.var(d) / len(d))
    p = 2 * (1 - norm.cdf(abs(DM)))

    return DM, p


# ==================================================
# Main
# ==================================================
if __name__ == "__main__":

    df = pd.read_csv(
        r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv"
    )

    df = df[df["timestamp"] >= "2023-01-01"]

    df["ret"] = np.log(df["close"]).diff()
    df["rv"] = np.abs(df["ret"])

    df.dropna(inplace=True)

    window = 1000

    results = rolling_oos_eval(
        returns=df["ret"].values,
        realized_vol=df["rv"].values,
        window=window,
        max_modes=10,
        gamma=0.2
    )

    logger.info("===== METRICS =====")

    for k in results.keys():
        # print(k, results[k].mean(), results[k].std())

        if k == "rv":
            continue
        mse = mean_squared_error(results["rv"], results[k]) * 100
        mae = mean_absolute_error(results["rv"], results[k]) * 100
        logger.info(f"{k.upper()} | MSE={mse:.6f} | MAE={mae:.6f}")

    


    DM1, p1 = diebold_mariano(
        results["rv"],
        results["garch_plain"],
        results["garch_ewt"]
    )

    DM2, p2 = diebold_mariano(
        results["rv"],
        results["garch_plain"],
        results["arima_low_garch_high"]
    )

    DM3, p3 = diebold_mariano(
        results["rv"],
        results["garch_plain"],
        results["svm_ewt"]
    )

    logger.info("===== DM TESTS =====")
    logger.info(f"GARCH vs EWT        : DM={DM1:.4f} p={p1:.6f}")
    logger.info(f"GARCH vs ARIMA+EWT  : DM={DM2:.4f} p={p2:.6f}")
    logger.info(f"GARCH vs SVM+EWT    : DM={DM3:.4f} p={p3:.6f}")

