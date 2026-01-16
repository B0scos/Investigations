import numpy as np
import pandas as pd

from numpy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import norm
from arch import arch_model

# ======================================================
# QLIKE LOSS
# ======================================================
def qlike(y, sigma2_hat, eps=1e-8):
    sigma2_hat = np.clip(sigma2_hat, eps, None)
    return y / sigma2_hat + np.log(sigma2_hat)


# ======================================================
# DIEBOLD–MARIANO TEST (QLIKE)
# ======================================================
def diebold_mariano_qlike(y, f1, f2):
    L1 = qlike(y, f1)
    L2 = qlike(y, f2)

    d = L1 - L2
    T = len(d)

    d_bar = np.mean(d)
    var_d = np.var(d, ddof=1)

    DM = d_bar / np.sqrt(var_d / T)
    p_value = 2 * (1 - norm.cdf(abs(DM)))

    return DM, p_value


# ======================================================
# GARCH(1,1)
# ======================================================
def fit_garch(series):
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


# ======================================================
# MINIMAL EWT (BOUNDARIES FIXED)
# ======================================================
def detect_boundaries(signal, max_modes=4):
    N = len(signal)
    spectrum = np.abs(fft(signal))[:N // 2]
    spectrum[0] = 0

    peaks, _ = find_peaks(spectrum, distance=(N // 2) // max_modes)
    peaks = np.sort(peaks[:max_modes])

    boundaries = 0.5 * (peaks[:-1] + peaks[1:])
    return boundaries / (N // 2)


def ewt_decompose(signal, max_modes=4):
    N = len(signal)
    freqs = np.abs(fftfreq(N))
    X = fft(signal)

    bounds = detect_boundaries(signal, max_modes)
    filters = []

    w0 = bounds[0]
    phi = (freqs <= w0).astype(float)
    filters.append(phi)

    for i in range(len(bounds) - 1):
        psi = ((freqs > bounds[i]) & (freqs <= bounds[i + 1])).astype(float)
        filters.append(psi)

    components = [np.real(ifft(X * f)) for f in filters]
    return components


# ======================================================
# ROLLING OOS EVALUATION
# ======================================================
def rolling_eval(returns, window=1000):
    sigma2_garch = []
    sigma2_ewt = []
    y = []

    for t in range(window, len(returns)):
        train = returns[t - window:t]

        # --- GARCH ---
        res_g = fit_garch(train)
        g_f = res_g.forecast(horizon=1).variance.values[-1, 0]

        # --- EWT → GARCH ---
        comps = ewt_decompose(train)
        vars_ = []

        for c in comps:
            if np.var(c) < 1e-8:
                continue
            res_c = fit_garch(c)
            v = res_c.forecast(horizon=1).variance.values[-1, 0]
            vars_.append(v)

        ewt_f = np.sum(vars_) if len(vars_) > 0 else np.nan

        sigma2_garch.append(g_f)
        sigma2_ewt.append(ewt_f)
        y.append(returns[t] ** 2)

    return (
        np.array(y),
        np.array(sigma2_garch),
        np.array(sigma2_ewt),
    )


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv")
    df["ret"] = np.log(df["close"]).diff()
    df = df.dropna()

    y, garch_f, ewt_f = rolling_eval(df["ret"].values)

    mask = ~np.isnan(ewt_f)
    y = y[mask]
    garch_f = garch_f[mask]
    ewt_f = ewt_f[mask]

    print("QLIKE GARCH     :", np.mean(qlike(y, garch_f)))
    print("QLIKE EWT-GARCH :", np.mean(qlike(y, ewt_f)))

    DM, p = diebold_mariano_qlike(y, garch_f, ewt_f)

    print("\nDiebold–Mariano (QLIKE)")
    print("DM statistic:", round(DM, 7))
    print("p-value     :", round(p, 7))
