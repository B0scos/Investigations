import numpy as np
import pandas as pd
import scipy.signal as signal
from numpy.fft import fft, ifft, fftfreq
from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# Gilles beta
# ======================================================
def beta(x):
    x = np.clip(x, 0, 1)
    return x**4 * (35 - 84*x + 70*x**2 - 20*x**3)

# ======================================================
# Spectrum smoothing
# ======================================================
def smooth_spectrum(spec, window=7):
    return np.convolve(spec, np.ones(window)/window, mode="same")

# ======================================================
# Boundary detection (Gilles Algorithm 1)
# ======================================================
def detect_boundaries(x):
    N = len(x)
    spectrum = np.abs(fft(x))[:N // 2]
    spectrum = smooth_spectrum(spectrum)

    peaks, _ = signal.find_peaks(spectrum)

    if len(peaks) < 2:
        raise ValueError("Not enough spectral peaks")

    boundaries = [(peaks[i] + peaks[i+1]) / 2 for i in range(len(peaks)-1)]
    boundaries = np.array(boundaries) / N  # normalize frequency

    return boundaries

# ======================================================
# EWT filter bank (Gilles Eq. 13–18)
# ======================================================
def build_ewt_filters(freqs, boundaries, gamma=0.25):
    filters = []
    abs_f = np.abs(freqs)

    # Scaling function
    w1 = boundaries[0]
    phi = np.zeros_like(freqs)

    for i, w in enumerate(abs_f):
        if w <= (1 - gamma) * w1:
            phi[i] = 1
        elif (1 - gamma) * w1 < w <= (1 + gamma) * w1:
            phi[i] = np.cos(
                np.pi/2 * beta((w - (1 - gamma)*w1) / (2*gamma*w1))
            )

    filters.append(phi)

    # Wavelets
    for n in range(len(boundaries) - 1):
        wn, wnp1 = boundaries[n], boundaries[n+1]
        psi = np.zeros_like(freqs)

        for i, w in enumerate(abs_f):
            if (1 + gamma)*wn <= w <= (1 - gamma)*wnp1:
                psi[i] = 1
            elif (1 - gamma)*wnp1 <= w <= (1 + gamma)*wnp1:
                psi[i] = np.cos(
                    np.pi/2 * beta((w - (1 - gamma)*wnp1) / (2*gamma*wnp1))
                )
            elif (1 - gamma)*wn <= w <= (1 + gamma)*wn:
                psi[i] = np.sin(
                    np.pi/2 * beta((w - (1 - gamma)*wn) / (2*gamma*wn))
                )

        filters.append(psi)

    return filters

# ======================================================
# EWT decomposition
# ======================================================
def ewt_decompose(x, gamma=0.25):
    N = len(x)
    freqs = fftfreq(N)
    X = fft(x)

    boundaries = detect_boundaries(x)
    filters = build_ewt_filters(freqs, boundaries, gamma)

    components = []
    for f in filters:
        components.append(np.real(ifft(X * f)))

    return components

# ======================================================
# GARCH one-step forecast
# ======================================================
def garch_forecast(train):
    model = arch_model(
        train,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=True
    )
    res = model.fit(disp="off")
    fcast = res.forecast(horizon=1, reindex=False)
    return np.sqrt(fcast.variance.values[-1, 0])

# ======================================================
# Rolling OOS evaluation
# ======================================================
def rolling_evaluation(returns, rv, start=500):
    garch_pred = []
    ewt_garch_pred = []

    for t in range(start, len(returns)):
        train = returns[:t]

        # ----- Plain GARCH -----
        garch_pred.append(garch_forecast(train))

        # ----- EWT → GARCH -----
        comps = ewt_decompose(train)
        sigmas = []

        for c in comps:
            if np.std(c) < 1e-8:
                continue
            sigmas.append(garch_forecast(c))

        sigma_total = np.sqrt(np.sum(np.array(sigmas)**2))
        ewt_garch_pred.append(sigma_total)

    return (
        np.array(garch_pred),
        np.array(ewt_garch_pred),
        rv[start:]
    )

# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":

    df = pd.read_csv(
        r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv"
    )

    df["ret"] = np.log(df["close"]).diff() * 100
    df = df.dropna()

    returns = df["ret"].values
    rv = np.abs(returns)

    garch_p, ewt_p, rv_test = rolling_evaluation(
        returns, rv, start=600
    )

    print("\n===== OUT-OF-SAMPLE RESULTS =====")
    print("Plain GARCH")
    print("  MAE :", mean_absolute_error(rv_test, garch_p))
    print("  MSE :", mean_squared_error(rv_test, garch_p))

    print("\nEWT → GARCH")
    print("  MAE :", mean_absolute_error(rv_test, ewt_p))
    print("  MSE :", mean_squared_error(rv_test, ewt_p))
