import numpy as np
import pandas as pd
import scipy.fft as fft
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# ================= CONFIG =================
CSV_PATH = r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv"
PRICE_COL = "close"
DATE_COL = "timestamp"

TRAIN_RATIO = 0.9
ARIMA_ORDER = (1, 0, 1)
GARCH_P, GARCH_Q = 1, 1

# ================= METRICS =================
def qlike(y_true, y_pred_var):
    return np.mean(np.log(y_pred_var) + y_true**2 / y_pred_var)

# ================= DATA =================
def load_returns():
    df = pd.read_csv(CSV_PATH, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL)
    r = np.log(df[PRICE_COL]).diff().dropna()
    return r.values

# ================= EWT DENOISING =================
def ewt_denoise(x, alpha=0.25):
    N = len(x)
    X = fft.fft(x)
    freqs = fft.fftfreq(N)
    cutoff = alpha * (0.5 * N)
    X[np.abs(freqs) > cutoff] = 0
    return np.real(fft.ifft(X))

# ================= FORECAST FUNCTIONS =================
def forecast_arima_garch(train, test_horizon):
    arima = ARIMA(train, order=ARIMA_ORDER).fit()
    resid = train - arima.fittedvalues
    garch = arch_model(resid, mean="Zero", vol="GARCH",
                       p=GARCH_P, q=GARCH_Q, dist="t")
    garch_fit = garch.fit(disp="off")
    garch_forecast = garch_fit.forecast(horizon=test_horizon).variance.values[-1]
    return garch_forecast

def forecast_plain_garch(train, test_horizon):
    garch = arch_model(train, mean="Zero", vol="GARCH",
                       p=GARCH_P, q=GARCH_Q, dist="t")
    fit = garch.fit(disp="off")
    return fit.forecast(horizon=test_horizon).variance.values[-1]

def forecast_plain_arima(train, test_horizon):
    arima = ARIMA(train, order=ARIMA_ORDER).fit()
    fcst_mean = arima.forecast(test_horizon)
    return fcst_mean**2  # variance proxy

# ================= MAIN =================
def main():
    returns = load_returns()
    n = len(returns)
    split = int(n*TRAIN_RATIO)
    train, test = returns[:split], returns[split:]
    H = len(test)

    # ------------------ FORECASTS ------------------
    baseline_var = forecast_arima_garch(train, H)
    hybrid_var = forecast_arima_garch(ewt_denoise(train, alpha=0.25), H)
    garch_var = forecast_plain_garch(train, H)
    arima_var = forecast_plain_arima(train, H)

    # ------------------ RESULTS ------------------
    print("\n===== VOLATILITY FORECAST RESULTS =====")
    print("Baseline ARIMA-GARCH:")
    print("  QLIKE :", qlike(test, baseline_var))
    print("  MSE   :", mean_squared_error(test**2, baseline_var))

    print("\nHybrid EWT + ARIMA-GARCH:")
    print("  QLIKE :", qlike(test, hybrid_var))
    print("  MSE   :", mean_squared_error(test**2, hybrid_var))

    print("\nPlain GARCH:")
    print("  QLIKE :", qlike(test, garch_var))
    print("  MSE   :", mean_squared_error(test**2, garch_var))

    print("\nPlain ARIMA (variance proxy):")
    print("  QLIKE :", qlike(test, arima_var))
    print("  MSE   :", mean_squared_error(test**2, arima_var))

if __name__ == "__main__":
    main()
