# ============================================================
# MODWT–ARIMA–GARCH Volatility Forecasting (Paper-faithful)
# Adapted to BTC data
# ============================================================

import numpy as np
import pandas as pd
import pywt
import warnings

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG (paper-aligned)
# ============================================================
CSV_PATH   = r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv"
PRICE_COL = "close"
DATE_COL  = "timestamp"

VOL_WINDOW  = 30
TRAIN_RATIO = 0.90

WAVELET = "db14"      # Best Localized (paper)
LEVEL   = 1           # J = 1 (paper uses shallow decomposition)

ARIMA_ORDER = (1, 1, 1)  # same order reported in paper

# ============================================================
# Metrics
# ============================================================
def rmse(y, yhat):
    return np.sqrt(mean_squared_error(y, yhat))

# ============================================================
# Load & preprocess data
# ============================================================
df = pd.read_csv(CSV_PATH, parse_dates=[DATE_COL])
df = df[df[DATE_COL] >= "2020-01-01"]
df = df.sort_values(DATE_COL)

# Log-returns
df["r"] = np.log(df[PRICE_COL] / df[PRICE_COL].shift(1))

# Volatility (rolling std)
df["vol"] = df["r"].rolling(VOL_WINDOW).std()

df.dropna(inplace=True)

vol = df["vol"].values

# ============================================================
# Train / Test split (90% / 10%)
# ============================================================
split = int(len(vol) * TRAIN_RATIO)

vol_train = vol[:split]
vol_test  = vol[split:]

H = len(vol_test)

# ============================================================
# 1. ARIMA-DIRECT (baseline)
# ============================================================
arima_direct = ARIMA(vol_train, order=ARIMA_ORDER).fit()
arima_direct_fcst = arima_direct.forecast(H)

# ============================================================
# 2. GARCH-DIRECT (baseline, on volatility)
# ============================================================
garch_direct = arch_model(
    vol_train,
    mean="Zero",
    vol="GARCH",
    p=1,
    q=1,
    dist="normal"
)

garch_direct_fit = garch_direct.fit(disp="off")
garch_direct_var = garch_direct_fit.forecast(horizon=H).variance.values[-1]
garch_direct_fcst = np.sqrt(garch_direct_var)

# ============================================================
# 3. MODWT decomposition of volatility
# ============================================================
# SWT = MODWT equivalent (no downsampling)
coeffs = pywt.swt(vol_train, wavelet=WAVELET, level=LEVEL)

# Approximation coefficients only (CA_J)
CA = coeffs[-1][0]

# ============================================================
# 4. MODWT + ARIMA
# ============================================================
arima_ca = ARIMA(CA, order=ARIMA_ORDER).fit()
modwt_arima_fcst = arima_ca.forecast(H)

# ============================================================
# 5. MODWT + GARCH
# ============================================================
garch_ca = arch_model(
    CA,
    mean="Zero",
    vol="GARCH",
    p=1,
    q=1,
    dist="normal"
)

garch_ca_fit = garch_ca.fit(disp="off")
garch_ca_var = garch_ca_fit.forecast(horizon=H).variance.values[-1]
modwt_garch_fcst = np.sqrt(garch_ca_var)

# ============================================================
# Evaluation (paper-style)
# ============================================================
results = pd.DataFrame({
    "Model": [
        "ARIMA-DIRECT",
        "GARCH-DIRECT",
        "MODWT-ARIMA (Bl14)",
        "MODWT-GARCH (Bl14)"
    ],
    "RMSE": [
        rmse(vol_test, arima_direct_fcst),
        rmse(vol_test, garch_direct_fcst),
        rmse(vol_test, modwt_arima_fcst),
        rmse(vol_test, modwt_garch_fcst),
    ]
})

print(results.sort_values("RMSE"))
