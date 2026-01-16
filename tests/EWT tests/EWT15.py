#!/usr/bin/env python3
"""
Volatility forecasting pipeline:
- HAR-RV baseline on log(RV)
- GARCH(1,1) on returns
- EWT decomposition on RETURNS -> per-component RV -> sum forecasts
- Non-negative linear stacking (weights learned on validation)
- Proper rolling one-step-ahead forecasts with no lookahead
- Metrics on volatility: MSE and MAE
"""

import argparse
import logging
import warnings
import traceback
from typing import List

import numpy as np
import pandas as pd
from numpy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.ar_model import AutoReg
from arch import arch_model
from scipy import stats

warnings.filterwarnings("ignore")

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# -------------------------
# Utilities / defensive
# -------------------------
EPS = 1e-12


def safe_sqrt(x):
    return np.sqrt(np.clip(x, 0.0, None))


# -------------------------
# EWT-related functions (apply to RETURNS)
# -------------------------
def smooth_beta(x):
    x = np.clip(x, 0.0, 1.0)
    return x ** 4 * (35 - 84 * x + 70 * x ** 2 - 20 * x ** 3)


def detect_boundaries(signal, max_modes=5):
    """
    Detect spectral peaks on the provided 1D real signal.
    Returns boundaries (normalized freq in cycles/sample) to pass to ewt_decompose.
    This should be called only on a training set (no lookahead).
    """
    N = len(signal)
    if N < 16:
        raise ValueError("Signal too short for boundary detection")

    spectrum = np.abs(fft(signal))[: N // 2]
    spectrum[0] = 0.0
    peaks, _ = find_peaks(spectrum, distance=max(1, (N // 2) // max(1, max_modes)))
    if len(peaks) < 2:
        raise RuntimeError("Not enough spectral peaks for EWT boundaries")

    # take strongest peaks up to max_modes, then sort by frequency index
    peaks = peaks[np.argsort(spectrum[peaks])[::-1]]
    peaks = np.sort(peaks[:max_modes])
    freqs = peaks / float(N)
    boundaries = 0.5 * (freqs[:-1] + freqs[1:])
    log.info(f"Detected {len(peaks)} peaks -> {len(boundaries)} boundaries")
    return boundaries


def ewt_decompose(signal, boundaries: List[float], gamma=0.2):
    """
    Decompose a real 1D signal into band-limited components using
    the supplied boundaries (normalized frequencies). Returns list of components
    of same length as signal. Boundaries must be computed from training data only.
    """
    N = len(signal)
    if N < 8:
        raise ValueError("Signal too short for EWT decomposition")

    freqs = np.abs(fftfreq(N))
    fft_sig = fft(signal)
    filters = []

    b = np.sort(np.asarray(boundaries))
    b = np.clip(b, 1e-12, 0.5 - 1e-12)

    # low-pass
    w1 = b[0]
    phi = np.zeros(N)
    low = freqs <= (1 - gamma) * w1
    trans = ((1 - gamma) * w1 < freqs) & (freqs <= (1 + gamma) * w1)
    phi[low] = 1.0
    phi[trans] = np.cos(
        (np.pi / 2) * smooth_beta((freqs[trans] - (1 - gamma) * w1) / (2 * gamma * w1))
    )
    filters.append(phi)

    # bandpass interiors
    for w0, w1 in zip(b[:-1], b[1:]):
        psi = np.zeros(N)
        band = ((1 + gamma) * w0 <= freqs) & (freqs <= (1 - gamma) * w1)
        psi[band] = 1.0
        up = ((1 - gamma) * w1 < freqs) & (freqs <= (1 + gamma) * w1)
        psi[up] = np.cos(
            (np.pi / 2) * smooth_beta((freqs[up] - (1 - gamma) * w1) / (2 * gamma * w1))
        )
        down = ((1 - gamma) * w0 <= freqs) & (freqs <= (1 + gamma) * w0)
        psi[down] = np.sin(
            (np.pi / 2) * smooth_beta((freqs[down] - (1 - gamma) * w0) / (2 * gamma * w0))
        )
        filters.append(psi)

    # high-pass
    wlast = b[-1]
    psi_last = np.zeros(N)
    high = freqs >= (1 + gamma) * wlast
    trans_last = ((1 - gamma) * wlast <= freqs) & (freqs <= (1 + gamma) * wlast)
    psi_last[high] = 1.0
    psi_last[trans_last] = np.sin(
        (np.pi / 2) * smooth_beta((freqs[trans_last] - (1 - gamma) * wlast) / (2 * gamma * wlast))
    )
    filters.append(psi_last)

    comps = []
    for f in filters:
        c = np.real(ifft(fft_sig * f))
        comps.append(c)
    return comps


# -------------------------
# Realized variance from returns (daily RV proxy)
# -------------------------
def realized_variance_from_returns(returns: np.ndarray, intraday_window: int = 1):
    """
    Given returns (daily log returns or high-freq returns), build realized variance series.
    If intraday_window==1 uses daily squared returns. If >1 uses rolling sum of squares.
    """
    s = pd.Series(returns).astype(float)
    if intraday_window <= 1:
        rv = s ** 2
    else:
        rv = s.rolling(intraday_window, min_periods=1).apply(lambda x: np.sum(x ** 2), raw=True)
    return rv.values


# -------------------------
# HAR-RV (log-linear) fitting / forecasting
# -------------------------
def har_features(rv_series: np.ndarray, idx: int, weekly=5, monthly=22):
    """
    Build HAR features at time idx for forecasting idx+1 using only past data up to idx.
    Returns feature vector [1, log(rv_t), log(rv_week), log(rv_month)]
    If insufficient history, returns None.
    """
    if idx < 0:
        return None
    # require at least monthly points
    start = max(0, idx - monthly + 1)
    if idx - 0 < 0:
        return None
    rv_t = rv_series[idx]
    # weekly mean: last 'weekly' days up to idx
    wstart = max(0, idx - weekly + 1)
    rv_week_mean = np.nanmean(rv_series[wstart: idx + 1])
    rv_month_mean = np.nanmean(rv_series[start: idx + 1])
    if np.isnan(rv_t) or np.isnan(rv_week_mean) or np.isnan(rv_month_mean):
        return None
    return np.array([1.0, np.log(rv_t + EPS), np.log(rv_week_mean + EPS), np.log(rv_month_mean + EPS)])


def har_fit_predict(rv_train: np.ndarray, rv_window_end_idx: int):
    """
    Fit HAR on rv_train[0:rv_window_end_idx+1] to forecast rv at rv_window_end_idx+1.
    Returns variance forecast (not volatility).
    """
    # build X,y from available history up to rv_window_end_idx-1 for predicting next
    X = []
    y = []
    # we need observations where next day's rv is available
    for t in range( max(0, rv_window_end_idx - 5000), rv_window_end_idx ):  # optional cap for speed
        feat = har_features(rv_train, t)
        if feat is None:
            continue
        # label is rv at t+1 if exists within training slice
        if t + 1 >= len(rv_train):
            break
        if np.isnan(rv_train[t + 1]):
            continue
        X.append(feat)
        y.append(np.log(rv_train[t + 1] + EPS))
    if len(X) < 10:
        # insufficient data: fallback to persistence
        return float(rv_train[rv_window_end_idx]) if rv_window_end_idx < len(rv_train) else float(rv_train[-1])
    X = np.vstack(X)
    y = np.asarray(y)
    lr = LinearRegression()
    lr.fit(X, y)
    # predict for rv_window_end_idx -> forecasting next
    feat_now = har_features(rv_train, rv_window_end_idx)
    if feat_now is None:
        return float(rv_train[rv_window_end_idx])
    logpred = lr.predict(feat_now.reshape(1, -1))[0]
    var_pred = float(max(np.exp(logpred) - EPS, EPS))
    return var_pred


# -------------------------
# Simple AR(1) forecast for a 1D series (used for component RV)
# -------------------------
def ar1_forecast(series: np.ndarray):
    """
    Fit AR(1) (AutoReg) on series and return one-step ahead forecast.
    If fit fails or insufficient data, fallback to last value.
    """
    try:
        if np.sum(np.isfinite(series)) < 10:
            return float(series[-1])
        # drop NaNs for fitting
        s = pd.Series(series).dropna().astype(float)
        if len(s) < 10:
            return float(series[-1])
        model = AutoReg(s, lags=1, old_names=False).fit()
        f = model.predict(start=len(s), end=len(s))[0]
        return float(max(f, 0.0))
    except Exception as e:
        log.debug(f"AR1 fit failed: {e}")
        return float(series[-1])


# -------------------------
# GARCH(1,1) forecast on returns -> variance forecast
# -------------------------
def garch_forecast_from_returns(returns_window: np.ndarray):
    """
    Fit GARCH(1,1) on returns_window and return 1-step ahead variance forecast.
    Falls back to sample variance or last squared return on failure.
    """
    try:
        if len(returns_window) < 50:
            # too short
            return float(np.nanvar(returns_window))
        am = arch_model(returns_window, mean="Zero", vol="GARCH", p=1, q=1, dist="t", rescale=False)
        res = am.fit(disp="off")
        vf = res.forecast(horizon=1).variance
        # handle different output shapes
        try:
            val = vf.values[-1, 0]
        except Exception:
            val = np.asarray(vf)[-1, 0]
        return float(max(val, EPS))
    except Exception as e:
        log.debug(f"GARCH fit failed: {e}")
        # fallback
        return float(max(np.nanvar(returns_window), EPS))


# -------------------------
# EWT-based variance forecast: decompose returns, compute component RV series, forecast each component variance, sum
# -------------------------
def ewt_variance_forecast(returns_window: np.ndarray, boundaries: List[float], gamma=0.2):
    """
    Given a returns window (length T), decompose returns into components (length T),
    compute per-component daily variance series = component^2, fit AR(1) on each component variance
    and sum the one-step ahead forecasts (variance additivity).
    """
    try:
        comps = ewt_decompose(returns_window, boundaries, gamma)
    except Exception as e:
        log.warning(f"EWT decomposition failed: {e}")
        # fallback: use simple variance of returns
        return float(max(np.nanvar(returns_window), EPS))

    total_forecast = 0.0
    for c in comps:
        comp_rv = (np.asarray(c) ** 2).astype(float)
        # use last portion to fit AR(1)
        f = ar1_forecast(comp_rv)
        total_forecast += max(float(f), 0.0)
    return float(max(total_forecast, EPS))


# -------------------------
# Diebold-Mariano test (simplified)
# -------------------------
def diebold_mariano(e1, e2, h=1):
    """
    Two-sided Diebold-Mariano test comparing two forecast errors (on same holdout).
    Uses Harvey et al. small-sample correction. Returns (dm_stat, p_value).
    e1, e2: series of losses (squared or absolute) for model1 and model2.
    """
    d = e1 - e2
    T = len(d)
    if T < 2:
        return np.nan, np.nan
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    dm = mean_d / np.sqrt(var_d / T)
    # two-sided p-value assuming normality
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return float(dm), float(p)


# -------------------------
# Pipeline runner
# -------------------------
def run_pipeline(csv_path: str,
                 date_col: str = None,
                 price_col: str = "close",
                 ret_col_name: str = "ret",
                 intraday_window: int = 1,
                 train_frac: float = 0.60,
                 val_frac: float = 0.20,
                 model_window: int = 750,
                 ewt_max_modes: int = 5,
                 ewt_gamma: float = 0.2):
    """
    Load data, create returns and RV proxy, and run the forecasting pipeline.
    """

    df = pd.read_csv(csv_path)
    df = df.copy()
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(date_col, inplace=True)
        df.reset_index(drop=True, inplace=True)

    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in CSV")

    # log returns
    df[ret_col_name] = np.log(df[price_col]).diff()
    df = df.dropna().reset_index(drop=True)
    returns = df[ret_col_name].values.astype(float)

    # realized variance series (daily proxy)
    rv = realized_variance_from_returns(returns, intraday_window=intraday_window)
    # ensure same length
    assert len(rv) == len(returns)

    N = len(rv)
    if N < 200:
        log.warning("Short series; some models may underperform due to lack of data")

    # chronological splits
    train_end = int(N * train_frac)
    val_end = int(N * (train_frac + val_frac))
    if val_end >= N:
        raise ValueError("Validation split too large; not enough test data")

    train_returns = returns[:train_end]
    val_returns = returns[train_end:val_end]
    test_returns = returns[val_end:]

    train_rv = rv[:train_end]
    val_rv = rv[train_end:val_end]
    test_rv = rv[val_end:]

    log.info(f"Data lengths: total={N}, train={len(train_rv)}, val={len(val_rv)}, test={len(test_rv)}")

    # 1) Learn EWT boundaries on TRAIN returns (no lookahead)
    try:
        train_returns_clean = train_returns[np.isfinite(train_returns)]
        boundaries = detect_boundaries(train_returns_clean, max_modes=ewt_max_modes)
    except Exception as e:
        log.warning(f"Boundary detection failed: {e}; falling back to single-band EWT (no decomposition)")
        boundaries = np.array([0.25])  # single split, trivial

    # -------------------------
    # Build validation forecasts for stacking (rolling one-step re-fit)
    # -------------------------
    log.info("Building validation one-step forecasts (this may take time)...")
    val_preds_har = []
    val_preds_garch = []
    val_preds_ewt = []
    val_truth = []

    # We'll use an expanding history that always uses data up to t-1 (start at train_end)
    # For t in [train_end .. val_end-1], we forecast rv[t] using data up to t-1
    for t in range(train_end, val_end):
        # build training window for models ending at t-1
        # we will use last 'model_window' observations up to t-1 for model fitting
        window_start = max(0, t - model_window)
        ret_hist = returns[window_start:t]   # up to t-1
        rv_hist = rv[window_start:t]
        if len(ret_hist) < 10 or len(rv_hist) < 10:
            # fallback to persistence
            val_preds_har.append(float(rv[t - 1]))
            val_preds_garch.append(float(np.nanvar(ret_hist) if len(ret_hist) > 0 else rv[t - 1]))
            val_preds_ewt.append(float(rv[t - 1]))
            val_truth.append(float(rv[t]))
            continue

        # HAR forecast
        try:
            har_f = har_fit_predict(rv_hist, len(rv_hist) - 1)
        except Exception as e:
            log.debug(f"HAR failed at val t={t}: {e}")
            har_f = float(rv_hist[-1])

        # GARCH forecast
        try:
            garch_f = garch_forecast_from_returns(ret_hist)
        except Exception as e:
            log.debug(f"GARCH val failure at t={t}: {e}")
            garch_f = float(max(np.nanvar(ret_hist), EPS))

        # EWT forecast (returns window)
        try:
            ewt_f = ewt_variance_forecast(ret_hist, boundaries, gamma=ewt_gamma)
        except Exception as e:
            log.debug(f"EWT val failure at t={t}: {e}")
            ewt_f = float(rv_hist[-1])

        val_preds_har.append(float(har_f))
        val_preds_garch.append(float(garch_f))
        val_preds_ewt.append(float(ewt_f))
        val_truth.append(float(rv[t]))

    val_preds = np.vstack([val_preds_har, val_preds_garch, val_preds_ewt]).T
    val_truth = np.asarray(val_truth)

    # -------------------------
    # Fit stacking weights on validation set (non-negative linear regression)
    # -------------------------
    log.info("Fitting non-negative stacking weights on validation set...")
    # safe: replace any nan/infs by small value
    val_preds = np.nan_to_num(val_preds, nan=EPS, posinf=1e9, neginf=EPS)
    val_truth = np.nan_to_num(val_truth, nan=EPS, posinf=1e9, neginf=EPS)

    # Since we work on variance, we fit on variance directly (could use log but keep linear on variance)
    try:
        stacker = LinearRegression(positive=True, fit_intercept=False)
        stacker.fit(val_preds, val_truth)
        weights = stacker.coef_
        # normalize weights to sum to 1 if sum>0 (keeps scale interpretable)
        s = weights.sum()
        if s > 0:
            weights = weights / s
    except Exception as e:
        log.warning(f"Stacker fitting failed: {e}; using equal weights")
        weights = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

    log.info(f"Stacking weights (HAR, GARCH, EWT) = {weights}")

    # -------------------------
    # Test: rolling one-step-ahead forecasts using only past (fixed weights)
    # -------------------------
    log.info("Running rolling one-step-ahead forecasts on TEST set...")
    y_true = []
    preds_persistence = []
    preds_har = []
    preds_garch = []
    preds_ewt = []
    preds_stack = []

    # For t in [val_end .. N-1], forecast rv[t] using data up to t-1
    for t in range(val_end, N):
        window_start = max(0, t - model_window)
        ret_hist = returns[window_start:t]   # up to t-1
        rv_hist = rv[window_start:t]
        if len(ret_hist) < 2:
            # record truth and fallback
            y_true.append(float(rv[t]))
            preds_persistence.append(float(rv[t - 1] if t - 1 >= 0 else rv[t]))
            preds_har.append(float(rv[t - 1] if t - 1 >= 0 else rv[t]))
            preds_garch.append(float(np.nanvar(ret_hist) if len(ret_hist) > 0 else rv[t]))
            preds_ewt.append(float(rv[t - 1] if t - 1 >= 0 else rv[t]))
            preds_stack.append(float(rv[t - 1] if t - 1 >= 0 else rv[t]))
            continue

        # true
        y_true.append(float(rv[t]))

        # persistence
        preds_persistence.append(float(rv_hist[-1]))

        # HAR
        try:
            har_f = har_fit_predict(rv_hist, len(rv_hist) - 1)
        except Exception as e:
            log.debug(f"HAR test failure at t={t}: {e}")
            har_f = float(rv_hist[-1])

        # GARCH
        try:
            garch_f = garch_forecast_from_returns(ret_hist)
        except Exception as e:
            log.debug(f"GARCH test failure at t={t}: {e}")
            garch_f = float(max(np.nanvar(ret_hist), EPS))

        # EWT
        try:
            ewt_f = ewt_variance_forecast(ret_hist, boundaries, gamma=ewt_gamma)
        except Exception as e:
            log.debug(f"EWT test failure at t={t}: {e}")
            ewt_f = float(rv_hist[-1])

        preds_har.append(float(har_f))
        preds_garch.append(float(garch_f))
        preds_ewt.append(float(ewt_f))

        # stacked combination using fixed weights
        comb = weights[0] * har_f + weights[1] * garch_f + weights[2] * ewt_f
        preds_stack.append(float(max(comb, EPS)))

    # Convert to volatility for evaluation
    y_vol = safe_sqrt(np.clip(np.asarray(y_true, dtype=float), 0.0, None))
    p_persist_vol = safe_sqrt(np.clip(np.asarray(preds_persistence, dtype=float), 0.0, None))
    p_har_vol = safe_sqrt(np.clip(np.asarray(preds_har, dtype=float), 0.0, None))
    p_garch_vol = safe_sqrt(np.clip(np.asarray(preds_garch, dtype=float), 0.0, None))
    p_ewt_vol = safe_sqrt(np.clip(np.asarray(preds_ewt, dtype=float), 0.0, None))
    p_stack_vol = safe_sqrt(np.clip(np.asarray(preds_stack, dtype=float), 0.0, None))

    # -------------------------
    # Metrics
    # -------------------------
    def report(name, true_v, pred_v):
        mse = mean_squared_error(true_v, pred_v)
        mae = mean_absolute_error(true_v, pred_v)
        log.info(f"{name:12s} | MSE={mse:.6f} | MAE={mae:.6f}")
        return mse, mae

    log.info("====== OOS RESULTS (volatility scale) ======")
    res = {}
    res['persistence'] = report("Persistence", y_vol, p_persist_vol)
    res['HAR'] = report("HAR-RV", y_vol, p_har_vol)
    res['GARCH'] = report("GARCH", y_vol, p_garch_vol)
    res['EWT'] = report("EWT-sum", y_vol, p_ewt_vol)
    res['STACK'] = report("STACK", y_vol, p_stack_vol)

    # Diebold-Mariano: compare STACK vs best baseline by MAE
    maes = {
        'persistence': res['persistence'][1],
        'HAR': res['HAR'][1],
        'GARCH': res['GARCH'][1],
        'EWT': res['EWT'][1]
    }
    best_baseline = min(maes, key=maes.get)
    log.info(f"Best baseline by MAE: {best_baseline}")

    # compute loss series (absolute errors) for DM test
    loss_stack = np.abs(y_vol - p_stack_vol)
    loss_base = np.abs(y_vol - {
        'persistence': p_persist_vol,
        'HAR': p_har_vol,
        'GARCH': p_garch_vol,
        'EWT': p_ewt_vol
    }[best_baseline])

    dm_stat, dm_p = diebold_mariano(loss_stack, loss_base)
    log.info(f"Diebold-Mariano (STACK vs {best_baseline}): DM={dm_stat:.4f} | p-value={dm_p:.4f}")

    # return forecasts and metrics
    out = {
        'y_vol': y_vol,
        'predictions': {
            'persistence': p_persist_vol,
            'HAR': p_har_vol,
            'GARCH': p_garch_vol,
            'EWT': p_ewt_vol,
            'STACK': p_stack_vol
        },
        'weights': weights,
        'metrics': res,
        'dm': (dm_stat, dm_p)
    }
    return out


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Volatility forecasting pipeline")
    parser.add_argument("--data", required=True, help="Path to CSV with price series")
    parser.add_argument("--price-col", default="close", help="Price column name")
    parser.add_argument("--date-col", default=None, help="Date column name (optional)")
    parser.add_argument("--intraday-window", type=int, default=1, help="Intraday window for RV")
    parser.add_argument("--model-window", type=int, default=750, help="Window length for model refits")
    parser.add_argument("--ewt-modes", type=int, default=5, help="Max EWT modes")
    args = parser.parse_args()

    try:
        result = run_pipeline(
            csv_path=args.data,
            date_col=args.date_col,
            price_col=args.price_col,
            intraday_window=args.intraday_window,
            model_window=args.model_window,
            ewt_max_modes=args.ewt_modes,
        )
        log.info("Pipeline finished successfully.")
    except Exception as e:
        log.error(f"Pipeline failed: {e}\n{traceback.format_exc()}")
        raise
