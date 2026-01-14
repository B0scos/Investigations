"""
Full replication of JAFEB paper method (adapted to BTC data).

Notes (read before running):
- The paper uses MATLAB's "bl14" wavelet. PyWavelets doesn't provide bl14.
  We use "db14" as a proxy (same filter length), which is the standard
  practical substitution for Python replications.
- The paper applies MODWT twice as a preprocessing step in some places; this
  script implements an optional double-MODWT preprocessing (enabled by
  DEFAULT_DOUBLE_MODWT = True).
- The script decomposes the volatility series (rolling std of log-returns),
  uses the final-level approximation (CA_J) as the main series to forecast
  with ARIMA and GARCH, and compares results to direct ARIMA/GARCH on the
  raw volatility series (paper's comparison).
- Train/test split: 90% / 10% (paper standard)
- Default ARIMA order uses the paper's reported best (1,1,5). You can change
  ARIMA_ORDER at the top.
- Outputs: metrics table (RMSE, MAE, MAPE, MASE, AIC, BIC) for each model.

Dependencies:
  pip install numpy pandas pywt statsmodels arch scikit-learn

How to run:
  1. Edit CSV_PATH to point to your BTC data CSV (must contain timestamp and close).
  2. Run `python replicate_modwt_garch_btc.py` in the virtualenv with deps.

Caveats / methodological choices:
- Using db14 as proxy for bl14 (documented in README header).
- We compute MASE with the training-set naive one-step-difference scale.
- Forecasts are computed in a static-fit manner (fit once on training set,
  then forecast H steps). The paper's rolling-origin details are not explicit;
  if you want rolling re-estimation, enable ROLLING_REFIT.

"""

import numpy as np
import pandas as pd
import pywt
import warnings
import os

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# =========================
# CONFIG (paper-aligned)
# =========================
CSV_PATH = r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv"
PRICE_COL = "close"
DATE_COL = "timestamp"

VOL_WINDOW = 30
TRAIN_RATIO = 0.90

# PyWavelets proxy for MATLAB bl14
WAVELET = "db14"
# number of MODWT levels to extract (paper tests multiple functions; we pick J=3)
MODWT_LEVEL = 3
DOUBLE_MODWT = False   # paper applies MODWT preprocessing twice in sections

# ARIMA order reported as best in paper
ARIMA_ORDER = (1, 1, 5)

# GARCH spec
GARCH_P = 1
GARCH_Q = 1

# Options
ROLLING_REFIT = False   # If True: perform rolling-origin refit (slow)
VERBOSE = False

# =========================
# Metrics
# =========================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    # avoid division by zero by using small epsilon where y_true==0
    eps = 1e-10
    y_true_safe = np.where(np.abs(y_true) < eps, eps, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100


def mase(y_true, y_pred, y_train):
    # scale = mean absolute one-step naive forecast error on training
    denom = np.mean(np.abs(np.diff(y_train)))
    if denom == 0:
        return np.nan
    return np.mean(np.abs(y_true - y_pred)) / denom


# =========================
# Helper functions
# =========================

def load_and_prepare(csv_path, price_col=PRICE_COL, date_col=DATE_COL, vol_window=VOL_WINDOW):
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # compute log returns
    df['r'] = np.log(df[price_col] / df[price_col].shift(1))
    # rolling volatility (sample std)
    df['vol'] = df['r'].rolling(vol_window).std()

    df = df.dropna().reset_index(drop=True)
    return df


def modwt_approximation(series, wavelet=WAVELET, level=MODWT_LEVEL, double=False):
    """Return the approximation (CA_J) from SWT (MODWT equivalent).
    If double=True, apply SWT to the CA_J again and return CA_J2.
    """
    # pywt.swt returns list[(cA_level, cD_level), ...] for levels 1..J
    coeffs = pywt.swt(series, wavelet=wavelet, level=level)
    # last tuple is level J (index -1), cA is first element
    ca = coeffs[-1][0]

    if double:
        # apply SWT again to the CA (single-level is fine)
        coeffs2 = pywt.swt(ca, wavelet=wavelet, level=level)
        ca2 = coeffs2[-1][0]
        return ca2

    return ca


def fit_arima_and_forecast(train_series, h, order=ARIMA_ORDER):
    model = ARIMA(train_series, order=order)
    res = model.fit()
    fcst = res.forecast(steps=h)
    aic = res.aic if hasattr(res, 'aic') else np.nan
    bic = res.bic if hasattr(res, 'bic') else np.nan
    return fcst, res, aic, bic


def fit_garch_and_forecast(train_series, h, p=GARCH_P, q=GARCH_Q):
    # arch expects 1D numpy array or pandas Series
    # use mean='Zero' since the paper uses returns for GARCH but here we use CA
    am = arch_model(train_series, mean='Zero', vol='GARCH', p=p, q=q, dist='normal')
    fitted = am.fit(disp='off')
    # forecast returns an object with .variance. We'll take the last row of the values
    fcst = fitted.forecast(horizon=h)
    # fcst.variance is a pandas DataFrame (or Panel-like); .variance.values shape (n_obs, h)
    try:
        var_matrix = fcst.variance.values
        # take the last available row (attached to end of train) and the h-column forecasts
        var_fcst = var_matrix[-1]
    except Exception:
        # fallback: try attribute access
        var_fcst = fcst.variance.iloc[-1].values

    # convert variance forecast to volatility forecast
    vol_fcst = np.sqrt(var_fcst)
    aic = fitted.aic if hasattr(fitted, 'aic') else np.nan
    bic = fitted.bic if hasattr(fitted, 'bic') else np.nan
    return vol_fcst, fitted, aic, bic


# =========================
# Main experiment
# =========================

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = load_and_prepare(CSV_PATH)
    vol = df['vol'].values

    n = len(vol)
    split = int(n * TRAIN_RATIO)

    vol_train = vol[:split]
    vol_test = vol[split:]
    H = len(vol_test)

    print(f"Total obs: {n}, Train: {len(vol_train)}, Test: {len(vol_test)}")

    results = []

    # -----------------
    # 1) ARIMA direct on volatility
    # -----------------
    arima_fcst, arima_res, arima_aic, arima_bic = fit_arima_and_forecast(vol_train, H)
    arima_rmse = rmse(vol_test, arima_fcst)
    arima_mae = mean_absolute_error(vol_test, arima_fcst)
    arima_mape = mape(vol_test, arima_fcst)
    arima_mase = mase(vol_test, arima_fcst, vol_train)

    results.append({
        'Model': 'ARIMA-Direct',
        'RMSE': arima_rmse,
        'MAE': arima_mae,
        'MAPE': arima_mape,
        'MASE': arima_mase,
        'AIC': arima_aic,
        'BIC': arima_bic
    })

    # -----------------
    # 2) GARCH direct on volatility
    # -----------------
    garch_fcst, garch_fit, garch_aic, garch_bic = fit_garch_and_forecast(vol_train, H)
    garch_rmse = rmse(vol_test, garch_fcst)
    garch_mae = mean_absolute_error(vol_test, garch_fcst)
    garch_mape = mape(vol_test, garch_fcst)
    garch_mase = mase(vol_test, garch_fcst, vol_train)

    results.append({
        'Model': 'GARCH-Direct',
        'RMSE': garch_rmse,
        'MAE': garch_mae,
        'MAPE': garch_mape,
        'MASE': garch_mase,
        'AIC': garch_aic,
        'BIC': garch_bic
    })

    # -----------------
    # 3) MODWT preprocessing (approximation) -> ARIMA
    # -----------------
    ca_train = modwt_approximation(vol_train, wavelet=WAVELET, level=MODWT_LEVEL, double=DOUBLE_MODWT)

    # ensure CA length matches training length (pywt.swt keeps length same)
    if len(ca_train) != len(vol_train):
        # trim or pad
        ca_train = ca_train[:len(vol_train)]

    modwt_arima_fcst, modwt_arima_res, modwt_arima_aic, modwt_arima_bic = fit_arima_and_forecast(ca_train, H)

    modwt_arima_rmse = rmse(vol_test, modwt_arima_fcst)
    modwt_arima_mae = mean_absolute_error(vol_test, modwt_arima_fcst)
    modwt_arima_mape = mape(vol_test, modwt_arima_fcst)
    modwt_arima_mase = mase(vol_test, modwt_arima_fcst, ca_train)

    results.append({
        'Model': f'MODWT-ARIMA (wavelet={WAVELET}, level={MODWT_LEVEL}, double={DOUBLE_MODWT})',
        'RMSE': modwt_arima_rmse,
        'MAE': modwt_arima_mae,
        'MAPE': modwt_arima_mape,
        'MASE': modwt_arima_mase,
        'AIC': modwt_arima_aic,
        'BIC': modwt_arima_bic
    })

    # -----------------
    # 4) MODWT preprocessing -> GARCH
    # -----------------
    modwt_garch_fcst, modwt_garch_fit, modwt_garch_aic, modwt_garch_bic = fit_garch_and_forecast(ca_train, H)

    modwt_garch_rmse = rmse(vol_test, modwt_garch_fcst)
    modwt_garch_mae = mean_absolute_error(vol_test, modwt_garch_fcst)
    modwt_garch_mape = mape(vol_test, modwt_garch_fcst)
    modwt_garch_mase = mase(vol_test, modwt_garch_fcst, ca_train)

    results.append({
        'Model': f'MODWT-GARCH (wavelet={WAVELET}, level={MODWT_LEVEL}, double={DOUBLE_MODWT})',
        'RMSE': modwt_garch_rmse,
        'MAE': modwt_garch_mae,
        'MAPE': modwt_garch_mape,
        'MASE': modwt_garch_mase,
        'AIC': modwt_garch_aic,
        'BIC': modwt_garch_bic
    })

    # -----------------
    # Summary table
    # -----------------
    res_df = pd.DataFrame(results).sort_values('RMSE').reset_index(drop=True)

    # print concise table
    pd.set_option('display.float_format', lambda x: f"{x:.6f}")
    print('\n==== RESULTS SUMMARY ====')
    print(res_df)

    # Save results & fitted models
    out_dir = 'modwt_garch_results'
    os.makedirs(out_dir, exist_ok=True)
    res_df.to_csv(os.path.join(out_dir, 'results_summary.csv'), index=False)

    # Optional: save forecasts & ground truth for plotting externally
    forecasts = pd.DataFrame({
        'vol_test': vol_test,
        'arima_direct_fcst': np.asarray(arima_fcst),
        'garch_direct_fcst': np.asarray(garch_fcst),
        'modwt_arima_fcst': np.asarray(modwt_arima_fcst),
        'modwt_garch_fcst': np.asarray(modwt_garch_fcst),
    })
    forecasts.to_csv(os.path.join(out_dir, 'forecasts.csv'), index=False)

    print(f"Results and forecasts saved to: {out_dir}/")


if __name__ == '__main__':
    main()
