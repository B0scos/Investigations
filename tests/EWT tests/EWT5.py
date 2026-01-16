import numpy as np
import pandas as pd

from numpy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks

from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==================================================
# Smooth transition function β(x)
# ==================================================
def beta(x):
    x = np.clip(x, 0.0, 1.0)
    return x**4 * (35 - 84*x + 70*x**2 - 20*x**3)


# ==================================================
# EWT boundary detection (Gilles-style)
# ==================================================
def detect_boundaries(signal, max_modes=6):
    N = len(signal)
    spectrum = np.abs(fft(signal))[: N // 2]

    spectrum[0] = 0.0  # remove DC

    peaks, _ = find_peaks(
        spectrum,
        distance=max(1, (N // 2) // max_modes)
    )

    if len(peaks) < 1:
        raise RuntimeError("No spectral peaks detected")

    # keep strongest peaks
    peaks = peaks[np.argsort(spectrum[peaks])[::-1]]
    peaks = np.sort(peaks[:max_modes])

    if len(peaks) < 2:
        raise RuntimeError("Not enough peaks for EWT")

    boundaries = [
        0.5 * (peaks[i] + peaks[i + 1])
        for i in range(len(peaks) - 1)
    ]

    return np.asarray(boundaries) / N


# ==================================================
# Empirical Wavelet Transform
# ==================================================
def ewt_decompose(signal, max_modes=6, gamma=0.25):
    N = len(signal)
    freqs = fftfreq(N)
    abs_freqs = np.abs(freqs)
    fft_signal = fft(signal)

    boundaries = detect_boundaries(signal, max_modes=max_modes)
    filters = []

    # ---- scaling function φ ----
    w1 = boundaries[0]
    phi = np.zeros_like(freqs)

    for i, w in enumerate(abs_freqs):
        if w <= (1 - gamma) * w1:
            phi[i] = 1.0
        elif (1 - gamma) * w1 < w <= (1 + gamma) * w1:
            phi[i] = np.cos(
                np.pi / 2 * beta((w - (1 - gamma) * w1) / (2 * gamma * w1))
            )

    filters.append(phi)

    # ---- wavelets ψ_n ----
    for n in range(len(boundaries) - 1):
        wn, wnp1 = boundaries[n], boundaries[n + 1]
        psi = np.zeros_like(freqs)

        for i, w in enumerate(abs_freqs):
            if (1 + gamma) * wn <= w <= (1 - gamma) * wnp1:
                psi[i] = 1.0
            elif (1 - gamma) * wnp1 <= w <= (1 + gamma) * wnp1:
                psi[i] = np.cos(
                    np.pi / 2 * beta(
                        (w - (1 - gamma) * wnp1) / (2 * gamma * wnp1)
                    )
                )
            elif (1 - gamma) * wn <= w <= (1 + gamma) * wn:
                psi[i] = np.sin(
                    np.pi / 2 * beta(
                        (w - (1 - gamma) * wn) / (2 * gamma * wn)
                    )
                )

        filters.append(psi)

    components = [
        np.real(ifft(fft_signal * filt))
        for filt in filters
    ]

    return components


# ==================================================
# GARCH fitting
# ==================================================
def fit_garch(series, p=1, q=1, mean="Zero", dist="t"):
    series = np.asarray(series)
    series = series[~np.isnan(series)]

    if len(series) < 50:
        raise RuntimeError("Series too short for GARCH")

    model = arch_model(
        series,
        mean=mean,
        vol="GARCH",
        p=p,
        q=q,
        dist=dist,
        rescale=True
    )

    res = model.fit(disp="off")
    return res


# ==================================================
# Rolling OOS evaluation
# ==================================================
def rolling_oos_eval(
    returns,
    realized_vol,
    window=500,
    max_modes=6,
    gamma=0.25
):
    garch_preds = []
    ewt_garch_preds = []
    rv_true = []

    for t in range(window, len(returns)):
        train = returns[t - window : t]

        # ---------- Plain GARCH ----------
        try:
            res_g = fit_garch(train)
            garch_forecast = np.sqrt(
                res_g.forecast(horizon=1).variance.values[-1, 0]
            )
        except:
            garch_forecast = np.nan

        # ---------- EWT → GARCH ----------
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

            # CORRECT aggregation
            ewt_forecast = (
                np.sqrt(np.sum(np.square(vols)))
                if len(vols) > 0 else np.nan
            )

        except:
            ewt_forecast = np.nan

        garch_preds.append(garch_forecast)
        ewt_garch_preds.append(ewt_forecast)
        rv_true.append(realized_vol[t])

    garch_preds = np.asarray(garch_preds)
    ewt_garch_preds = np.asarray(ewt_garch_preds)
    rv_true = np.asarray(rv_true)

    mask = ~np.isnan(garch_preds) & ~np.isnan(ewt_garch_preds)

    if np.sum(mask) == 0:
        raise RuntimeError("No valid OOS forecasts produced")

    return {
        "GARCH_MSE": mean_squared_error(rv_true[mask], garch_preds[mask]),
        "GARCH_MAE": mean_absolute_error(rv_true[mask], garch_preds[mask]),
        "EWT_GARCH_MSE": mean_squared_error(rv_true[mask], ewt_garch_preds[mask]),
        "EWT_GARCH_MAE": mean_absolute_error(rv_true[mask], ewt_garch_preds[mask]),
        "VALID_POINTS": int(np.sum(mask)),
    }


# ==================================================
# Main
# ==================================================
if __name__ == "__main__":
    df = pd.read_csv(
        r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv"
    )

    df = df[df["timestamp"] >= "2022-01-01"]

    df["ret"] = np.log(df["close"]).diff() * 100
    df["rv"] = df["ret"].abs()
    df = df.dropna()

    window = 1000
    print(f"Window size: {window}")

    results = rolling_oos_eval(
        returns=df["ret"].values,
        realized_vol=df["rv"].values,
        window=window,
        max_modes=6,
        gamma=0.3
    )

    print("\n===== ROLLING OOS RESULTS =====")
    for k, v in results.items():
        print(f"{k}: {v}")
