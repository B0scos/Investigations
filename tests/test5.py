import numpy as np
import pandas as pd
import scipy.fft as fft
import warnings

from scipy.signal import find_peaks
from arch import arch_model
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# ======================================================
# CONFIG
# ======================================================

CSV_PATH = r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv"
PRICE_COL = "close"
DATE_COL = "timestamp"

TRAIN_RATIO = 0.90
EWT_MIN_PEAKS = 3
GARCH_P, GARCH_Q = 1, 1

# ======================================================
# Metrics
# ======================================================

def rmse(y, yhat):
    return np.sqrt(mean_squared_error(y, yhat))

# ======================================================
# Data prep (RETURNS, not volatility)
# ======================================================

def load_returns():
    df = pd.read_csv(CSV_PATH, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL)

    r = np.log(df[PRICE_COL]).diff().dropna()
    r = r - r.mean()  # REQUIRED for EWT

    return r.values

# ======================================================
# True EWT implementation
# ======================================================

def ewt_boundaries(signal, min_peaks=3):
    N = len(signal)

    X = np.abs(fft.fft(signal))[:N // 2]
    X[0] = 0  # remove DC
    freqs = np.linspace(0, np.pi, N // 2)

    # Spectrum envelope peaks
    peaks, _ = find_peaks(X, distance=5)

    if len(peaks) < min_peaks:
        peaks = np.argsort(X)[-min_peaks:]

    peak_freqs = np.sort(freqs[peaks])

    # Midpoints between peaks
    bounds = 0.5 * (peak_freqs[:-1] + peak_freqs[1:])
    return bounds


def meyer_filter(freqs, w1, w2):
    H = np.zeros_like(freqs)

    def smooth(x):
        return x**4 * (35 - 84*x + 70*x**2 - 20*x**3)

    idx1 = (freqs >= w1) & (freqs <= w2)
    t = (freqs[idx1] - w1) / (w2 - w1)
    H[idx1] = smooth(t)

    H[freqs > w2] = 1
    return H


def ewt_decompose(signal):
    N = len(signal)
    X = fft.fft(signal)
    freqs = np.linspace(0, np.pi, N // 2 + 1)

    bounds = ewt_boundaries(signal)

    bands = []

    def mirror(H):
        full = np.zeros(N)
        full[:len(H)] = H
        full[len(H):] = H[-2:0:-1]
        return full

    # Low-pass
    H = meyer_filter(freqs, 0, bounds[0])
    bands.append(np.real(fft.ifft(X * mirror(H))))

    # Band-pass
    for i in range(len(bounds) - 1):
        H1 = meyer_filter(freqs, bounds[i], bounds[i + 1])
        H0 = meyer_filter(freqs, 0, bounds[i])
        bands.append(np.real(fft.ifft(X * mirror(H1 - H0))))

    # High-pass
    H = 1 - meyer_filter(freqs, 0, bounds[-1])
    bands.append(np.real(fft.ifft(X * mirror(H))))

    return bands

# ======================================================
# GARCH forecasting per band
# ======================================================

def forecast_band_variance(train, h):
    model = arch_model(train, mean="Zero",
                       vol="GARCH", p=GARCH_P, q=GARCH_Q)
    fit = model.fit(disp="off")
    return fit.forecast(horizon=h).variance.values[-1]

# ======================================================
# Main experiment
# ======================================================

def main():
    r = load_returns()
    n = len(r)
    split = int(n * TRAIN_RATIO)

    train, test = r[:split], r[split:]
    H = len(test)

    # -------------------------
    # Baseline GARCH (returns)
    # -------------------------
    base_model = arch_model(train, mean="Zero",
                            vol="GARCH", p=1, q=1)
    base_fit = base_model.fit(disp="off")
    base_var = base_fit.forecast(horizon=H).variance.values[-1]
    base_vol = np.sqrt(base_var)

    # -------------------------
    # EWT–GARCH
    # -------------------------
    bands = ewt_decompose(train)

    band_vars = []
    for b in bands:
        band_vars.append(forecast_band_variance(b, H))

    total_var = np.sum(band_vars, axis=0)
    ewt_vol = np.sqrt(total_var)

    # -------------------------
    # Proxy realized volatility
    # -------------------------
    realized_vol = np.abs(test)

    print("\n===== RESULTS =====")
    print("Baseline GARCH RMSE :", rmse(realized_vol, base_vol))
    print("EWT–GARCH RMSE      :", rmse(realized_vol, ewt_vol))


if __name__ == "__main__":
    main()
