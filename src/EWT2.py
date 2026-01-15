import numpy as np
import pandas as pd
from numpy.fft import fft, ifft, fftfreq
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# EWT CORE
# ============================================================

def beta(x):
    x = np.clip(x, 0, 1)
    return x**4 * (35 - 84*x + 70*x**2 - 20*x**3)

def detect_boundaries(signal, alpha=0.1):
    N = len(signal)
    spectrum = np.abs(fft(signal))[:N // 2]

    spec_norm = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())
    threshold = spec_norm.min() + alpha * (spec_norm.max() - spec_norm.min())

    idx = np.where(spec_norm >= threshold)[0]

    if len(idx) < 2:
        raise RuntimeError("EWT failed: not enough spectral structure")

    return np.array([(idx[i] + idx[i + 1]) / 2 for i in range(len(idx) - 1)])

def build_ewt_filters(freqs, boundaries, gamma=0.2):
    filters = []

    w1 = boundaries[0]
    phi = np.zeros_like(freqs)

    for i, w in enumerate(np.abs(freqs)):
        if w <= (1 - gamma) * w1:
            phi[i] = 1
        elif (1 - gamma) * w1 < w <= (1 + gamma) * w1:
            phi[i] = np.cos(np.pi / 2 * beta((w - (1 - gamma) * w1) / (2 * gamma * w1)))

    filters.append(phi)

    for n in range(len(boundaries) - 1):
        wn, wnp1 = boundaries[n], boundaries[n + 1]
        psi = np.zeros_like(freqs)

        for i, w in enumerate(np.abs(freqs)):
            if (1 + gamma) * wn <= w <= (1 - gamma) * wnp1:
                psi[i] = 1
            elif (1 - gamma) * wnp1 <= w <= (1 + gamma) * wnp1:
                psi[i] = np.cos(np.pi / 2 * beta((w - (1 - gamma) * wnp1) / (2 * gamma * wnp1)))
            elif (1 - gamma) * wn <= w <= (1 + gamma) * wn:
                psi[i] = np.sin(np.pi / 2 * beta((w - (1 - gamma) * wn) / (2 * gamma * wn)))

        filters.append(psi)

    return filters

def ewt_decompose(signal):
    N = len(signal)
    freqs = fftfreq(N)
    fft_signal = fft(signal)

    boundaries = detect_boundaries(signal)
    filters = build_ewt_filters(freqs, boundaries)

    components = []
    for filt in filters:
        comp = np.real(ifft(fft_signal * filt))
        if np.std(comp) > 1e-8:
            components.append(comp)

    return components

# ============================================================
# GARCH
# ============================================================

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
    res = model.fit(disp="off")
    return res.conditional_volatility

# ============================================================
# METRICS
# ============================================================

def qlike(y_true, y_pred):
    eps = 1e-8
    return np.mean(
        np.log(y_pred**2 + eps) +
        (y_true**2) / (y_pred**2 + eps)
    )

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    df = pd.read_csv(
        r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv"
    )

    df["ret"] = np.log(df["close"]).diff() 
    df.dropna(inplace=True)

    returns = df["ret"].values

    # --------------------------------------------------------
    # Realized volatility proxy (daily data)
    # --------------------------------------------------------
    df["RV"] = df["ret"] ** 2
    df.dropna(inplace=True)

    rv = df["RV"].values

    # --------------------------------------------------------
    # Baseline: Plain GARCH
    # --------------------------------------------------------
    garch_vol = fit_garch(df["ret"].values)
    garch_vol = garch_vol[-len(rv):]

    # --------------------------------------------------------
    # EWT → GARCH
    # --------------------------------------------------------
    components = ewt_decompose(returns)

    component_vols = []
    s = 0
    for comp in components:

        if np.min(comp) == np.max(comp):
            # Skip constant components
            continue


        vol = fit_garch(comp)
        component_vols.append(vol)

        s += 1

    print(f"numero : {s}")

    min_len = min(len(v) for v in component_vols)
    vol_matrix = np.column_stack([v[-min_len:] for v in component_vols])

    # Variance aggregation (CRITICAL)
    ewt_garch_vol = np.sqrt(np.sum(vol_matrix**2, axis=1))

    rv = rv[-min_len:]
    garch_vol = garch_vol[-min_len:]

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    print("\n===== VOLATILITY FORECAST COMPARISON =====")

    print("\nPlain GARCH vs RV")
    print("QLIKE :", qlike(rv, garch_vol))
    print("MSE   :", mean_squared_error(rv, garch_vol))
    print("MAE   :", mean_absolute_error(rv, garch_vol))

    print("\nEWT–GARCH vs RV")
    print("QLIKE :", qlike(rv, ewt_garch_vol))
    print("MSE   :", mean_squared_error(rv, ewt_garch_vol))
    print("MAE   :", mean_absolute_error(rv, ewt_garch_vol))
