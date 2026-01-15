import numpy as np
import pandas as pd

from numpy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import norm

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

    if len(peaks) < 2:
        raise RuntimeError("Not enough spectral peaks")

    peaks = peaks[np.argsort(spectrum[peaks])[::-1]]
    peaks = np.sort(peaks[:max_modes])

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

    # ---- Scaling function φ ----
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

    # ---- Wavelets ψ_n ----
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
def fit_garch(series):
    series = np.asarray(series)
    series = series[~np.isnan(series)]

    if len(series) < 50:
        raise RuntimeError("Series too short for GARCH")

    model = arch_model(
        series,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=True
    )

    return model.fit(disp="off")


# ==================================================
# Rolling OOS evaluation
# ==================================================
def rolling_oos_eval(
    returns,
    realized_vol,
    window,
    max_modes=6,
    gamma=0.25
):
    garch_preds = []
    ewt_preds = []
    rv_true = []

    for t in range(window, len(returns)):
        train = returns[t - window : t]

        # ---- Plain GARCH ----
        try:
            res_g = fit_garch(train)
            garch_f = np.sqrt(
                res_g.forecast(horizon=1).variance.values[-1, 0]
            )
        except:
            garch_f = np.nan

        # ---- EWT → GARCH ----
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

            ewt_f = (
                np.sqrt(np.sum(np.square(vols)))
                if len(vols) > 0 else np.nan
            )
        except:
            ewt_f = np.nan

        garch_preds.append(garch_f)
        ewt_preds.append(ewt_f)
        rv_true.append(realized_vol[t])

    garch_preds = np.asarray(garch_preds)
    ewt_preds = np.asarray(ewt_preds)
    rv_true = np.asarray(rv_true)

    mask = ~np.isnan(garch_preds) & ~np.isnan(ewt_preds)

    if np.sum(mask) == 0:
        raise RuntimeError("No valid OOS forecasts")

    return {
        "rv": rv_true[mask],
        "garch": garch_preds[mask],
        "ewt": ewt_preds[mask],
    }


# ==================================================
# Diebold–Mariano test
# ==================================================
def diebold_mariano(y, f1, f2, loss="mse", h=1):
    y, f1, f2 = map(np.asarray, (y, f1, f2))

    if loss == "mse":
        d = (y - f1)**2 - (y - f2)**2
    elif loss == "mae":
        d = np.abs(y - f1) - np.abs(y - f2)
    else:
        raise ValueError("loss must be 'mse' or 'mae'")

    T = len(d)
    d_bar = np.mean(d)

    gamma = np.array([
        np.var(d, ddof=0) if k == 0
        else np.cov(d[:-k], d[k:], bias=True)[0, 1]
        for k in range(h)
    ])

    var_d = gamma[0] + 2 * np.sum(gamma[1:])
    DM = d_bar / np.sqrt(var_d / T)

    p_value = 2 * (1 - norm.cdf(abs(DM)))

    return DM, p_value


def qlike(y, sigma2_hat, eps=1e-8):
    sigma2_hat = np.clip(sigma2_hat, eps, None)
    return y / sigma2_hat + np.log(sigma2_hat)

# ==================================================
# Main
# ==================================================
if __name__ == "__main__":
    df = pd.read_csv(
        r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv"
    )

    df = df[df["timestamp"] >= "2022-01-01"]

    df["ret"] = np.log(df["close"]).diff() #* 100
    df["rv"] = df["ret"] ** 2
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

    # ---- Metrics ----
    print("\n===== OOS METRICS =====")
    print("GARCH MSE:", mean_squared_error(results["rv"], results["garch"]))
    print("EWT–GARCH MSE:", mean_squared_error(results["rv"], results["ewt"]))

    print("GARCH MAE:", mean_absolute_error(results["rv"], results["garch"]))
    print("EWT–GARCH MAE:", mean_absolute_error(results["rv"], results["ewt"]))

    print("GARCH qlike:", np.mean(qlike(results["rv"], results["garch"])))
    print("EWT–GARCH qlike:", np.mean(qlike(results["rv"], results["ewt"])))


    # ---- Diebold–Mariano ----
    DM, p = diebold_mariano(
        y=results["rv"],
        f1=results["garch"],
        f2=results["ewt"],
        loss="mse",
        h=1
    )

    print("\n===== DIEBOLD–MARIANO TEST =====")
    print(f"DM statistic : {DM:.4f}")
    print(f"p-value      : {p:.4f}")
