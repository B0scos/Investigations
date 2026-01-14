import numpy as np
import pandas as pd
import scipy.fft as fft
import warnings

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ======================================================
# CONFIG
# ======================================================

CSV_PATH = r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv"
PRICE_COL = "close"
DATE_COL = "timestamp"

VOL_WINDOW = 30
TRAIN_RATIO = 0.90
EWT_ALPHA = 0.25

ARIMA_ORDER = (1, 1, 1)
GARCH_P, GARCH_Q = 1, 1

# ======================================================
# Metrics
# ======================================================

def rmse(y, yhat):
    return np.sqrt(mean_squared_error(y, yhat))

# ======================================================
# Data prep
# ======================================================

def load_volatility():
    df = pd.read_csv(CSV_PATH, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL)

    r = np.log(df[PRICE_COL] / df[PRICE_COL].shift(1)) 
    vol = r.rolling(VOL_WINDOW).std()

    return vol.dropna().values

# ======================================================
# EWT decomposition
# ======================================================

from scipy.signal import find_peaks

def ewt_decompose(signal, alpha=0.25, min_peaks=3):
    N = len(signal)

    # FFT
    X = np.abs(fft.fft(signal))
    freqs = fft.fftfreq(N)

    mask = freqs > 0
    X, freqs = X[mask], freqs[mask]

    # Normalize
    Xn = (X - X.min()) / (X.max() - X.min() + 1e-12)

    # --------------------------------------------------
    # 1. Peak detection (robust)
    # --------------------------------------------------
    peaks, props = find_peaks(Xn, height=alpha)

    # If too few peaks → relax threshold
    if len(peaks) < min_peaks:
        peaks, props = find_peaks(Xn, height=np.quantile(Xn, 0.75))

    # Still too few → fallback to strongest frequencies
    if len(peaks) < min_peaks:
        peaks = np.argsort(Xn)[-min_peaks:]

    peak_freqs = np.sort(freqs[peaks])

    # --------------------------------------------------
    # 2. Boundary construction
    # --------------------------------------------------
    bounds = 0.5 * (peak_freqs[:-1] + peak_freqs[1:])

    # Normalize to [0, π]
    bounds = np.pi * bounds / bounds.max()

    return build_ewt_bands(signal, bounds)


def build_ewt_bands(signal, bounds):
    N = len(signal)
    X = fft.fft(signal)
    freqs = np.linspace(0, np.pi, N // 2 + 1)

    bands = []

    def mirror(H):
        full = np.zeros(N)
        full[:len(H)] = H
        full[len(H):] = H[-2:0:-1]
        return full

    # Low-pass
    H = np.zeros_like(freqs)
    H[freqs <= bounds[0]] = 1
    bands.append(np.real(fft.ifft(X * mirror(H))))

    # Band-pass
    for i in range(len(bounds) - 1):
        H = np.zeros_like(freqs)
        H[(freqs >= bounds[i]) & (freqs <= bounds[i + 1])] = 1
        bands.append(np.real(fft.ifft(X * mirror(H))))

    # High-pass
    H = np.zeros_like(freqs)
    H[freqs >= bounds[-1]] = 1
    bands.append(np.real(fft.ifft(X * mirror(H))))

    return bands

# ======================================================
# Band diagnostics
# ======================================================

def band_is_heteroskedastic(x):
    try:
        pval = het_arch(x)[1]
        return pval < 0.05
    except:
        return False

# ======================================================
# Forecasting per band
# ======================================================

def forecast_band(train, h, use_garch):
    if use_garch:
        model = arch_model(train, mean="Zero", vol="GARCH",
                           p=GARCH_P, q=GARCH_Q)
        fit = model.fit(disp="off")
        f = fit.forecast(horizon=h).variance.values[-1]
        return np.sqrt(f)
    else:
        model = ARIMA(train, order=ARIMA_ORDER)
        fit = model.fit()
        return fit.forecast(h)

# ======================================================
# Main experiment
# ======================================================

def main():
    vol = load_volatility()
    n = len(vol)
    split = int(n * TRAIN_RATIO)

    train, test = vol[:split], vol[split:]
    H = len(test)

    # -------------------------
    # Baseline: GARCH
    # -------------------------
    base_model = arch_model(train, mean="Zero", vol="GARCH",
                            p=1, q=1)
    base_fit = base_model.fit(disp="off")
    base_fcst = np.sqrt(base_fit.forecast(horizon=H).variance.values[-1])

    # -------------------------
    # ABS-VM
    # -------------------------
    bands = ewt_decompose(train, alpha=EWT_ALPHA)

    band_forecasts = []

    for b in bands:
        use_garch = band_is_heteroskedastic(b)
        band_forecasts.append(forecast_band(b, H, use_garch))

    abs_vm_forecast = np.sum(band_forecasts, axis=0)

    # -------------------------
    # Results
    # -------------------------
    print("\n===== RESULTS =====")
    print("Baseline GARCH RMSE :", rmse(test, base_fcst))
    print("ABS-VM RMSE         :", rmse(test, abs_vm_forecast))

if __name__ == "__main__":
    main()
