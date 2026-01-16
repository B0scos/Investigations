import numpy as np
import pandas as pd
import logging
from arch import arch_model
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import norm
from numpy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks

# ======================================================
# CONFIG
# ======================================================

CSV_PATH = r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv"
WINDOW_TRAIN = 2000
RV_WINDOW = 5
EPS = 1e-8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ======================================================
# DATA
# ======================================================

def load_data(path):

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    price = df["close"]
    returns = np.log(price).diff().dropna()

    return price.iloc[1:], returns


# ======================================================
# REALIZED VOLATILITY
# ======================================================

def realized_volatility(returns, window):

    rv = returns.rolling(window).apply(
        lambda x: np.sqrt(np.sum(x**2)),
        raw=True
    )

    return rv.dropna()


# ======================================================
# HAR FEATURES
# ======================================================

def har_features(rv):

    df = pd.DataFrame({"rv": rv})

    df["d"] = df["rv"].shift(1)
    df["w"] = df["rv"].rolling(5).mean().shift(1)
    df["m"] = df["rv"].rolling(22).mean().shift(1)

    return df.dropna()


# ======================================================
# HAR MODEL
# ======================================================

def fit_har(train):

    X = train[["d", "w", "m"]].values
    y = np.log(train["rv"].values + EPS)

    X = np.column_stack([np.ones(len(X)), X])

    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    return beta


def predict_har(beta, row):

    x = np.array([1, row["d"], row["w"], row["m"]])

    log_pred = x @ beta

    return np.exp(log_pred)


# ======================================================
# GARCH
# ======================================================

def fit_garch(ret):

    model = arch_model(
        ret * 100,
        p=1,
        q=1,
        mean="Zero",
        vol="Garch",
        dist="t"
    )

    res = model.fit(disp="off")

    return res


def predict_garch(res):

    f = res.forecast(horizon=1)

    sigma = np.sqrt(f.variance.values[-1, 0])

    return sigma / 100


# ======================================================
# EWT (Residual Decomposition)
# ======================================================

def ewt_decompose(signal, max_modes=3):

    N = len(signal)

    spec = np.abs(fft(signal))[:N // 2]
    spec[0] = 0

    peaks, _ = find_peaks(spec, distance=N // 8)

    if len(peaks) < 2:
        return [signal]

    peaks = peaks[np.argsort(spec[peaks])][-max_modes:]
    peaks = np.sort(peaks)

    freqs = peaks / N
    bounds = (freqs[:-1] + freqs[1:]) / 2

    fft_sig = fft(signal)
    freq_axis = np.abs(fftfreq(N))

    comps = []

    prev = 0.0

    for b in list(bounds) + [0.5]:

        mask = (freq_axis >= prev) & (freq_axis < b)

        band = np.zeros(N)
        band[mask] = 1.0

        comps.append(np.real(ifft(fft_sig * band)))

        prev = b

    return comps


# ======================================================
# DIEBOLD MARIANO
# ======================================================

def diebold_mariano(e1, e2):

    d = e1 - e2

    dm = np.mean(d) / np.sqrt(np.var(d, ddof=1) / len(d))

    p = 2 * (1 - norm.cdf(abs(dm)))

    return dm, p


# ======================================================
# PIPELINE
# ======================================================

def run_pipeline(returns):

    rv = realized_volatility(returns, RV_WINDOW)
    har_df = har_features(rv)

    returns = returns.loc[har_df.index]

    preds_har = []
    preds_garch = []
    preds_ewt = []
    true_vals = []

    stack_X = []
    stack_y = []

    logging.info(f"Train window: {WINDOW_TRAIN}")
    logging.info(f"Test window : {len(har_df) - WINDOW_TRAIN}")

    for t in range(WINDOW_TRAIN, len(har_df)):

        train = har_df.iloc[t - WINDOW_TRAIN:t]
        test = har_df.iloc[t]

        # =========================
        # HAR
        # =========================

        beta = fit_har(train)

        har_pred = predict_har(beta, test)

        # =========================
        # EWT residual correction
        # =========================

        X_train = train[["d","w","m"]].values
        X_train = np.column_stack([np.ones(len(X_train)), X_train])

        log_har_train = X_train @ beta

        log_rv_train = np.log(train["rv"].values + EPS)

        residuals = log_rv_train - log_har_train

        comps = ewt_decompose(residuals)

        low_freq = comps[0]

        # linear drift extrapolation
        idx = np.arange(len(low_freq))
        a, b = np.polyfit(idx, low_freq, 1)

        next_corr = a * len(low_freq) + b

        har_ewt_pred = har_pred * np.exp(next_corr)

        # =========================
        # GARCH
        # =========================

        ret_train = returns.iloc[t - WINDOW_TRAIN:t]

        garch_fit = fit_garch(ret_train)
        garch_pred = predict_garch(garch_fit)

        # =========================
        # Residual stacking target
        # =========================

        residual_target = test["rv"] - har_ewt_pred

        stack_X.append([har_ewt_pred, garch_pred])
        stack_y.append(residual_target)

        preds_har.append(har_pred)
        preds_ewt.append(har_ewt_pred)
        preds_garch.append(garch_pred)
        true_vals.append(test["rv"])

        if (t - WINDOW_TRAIN) % 200 == 0:
            logging.info(f"OOS step {t - WINDOW_TRAIN}")

    stack_X = np.array(stack_X)
    stack_y = np.array(stack_y)

    preds_har = np.array(preds_har)
    preds_ewt = np.array(preds_ewt)
    preds_garch = np.array(preds_garch)
    true_vals = np.array(true_vals)

    # ======================================================
    # META STACKER (MAE optimal)
    # ======================================================

    meta = QuantileRegressor(
        quantile=0.5,
        alpha=0.0,
        solver="highs"
    )

    meta.fit(stack_X, stack_y)

    correction = meta.predict(stack_X)

    preds_stack = preds_ewt + correction

    # ======================================================
    # METRICS
    # ======================================================

    logging.info("====== OOS RESULTS (volatility scale) ======")

    def report(name, yhat):
        mse = mean_squared_error(true_vals, yhat)
        mae = mean_absolute_error(true_vals, yhat)
        logging.info(f"{name:<10} | MSE={mse:.6f} | MAE={mae:.6f}")
        return mae

    mae_har = report("HAR", preds_har)
    mae_ewt = report("HAR+EWT", preds_ewt)
    mae_stack = report("STACK", preds_stack)

    # ======================================================
    # DM TEST
    # ======================================================

    e_har = np.abs(true_vals - preds_har)
    e_stack = np.abs(true_vals - preds_stack)

    dm, p = diebold_mariano(e_stack, e_har)

    logging.info(f"DM (STACK vs HAR) = {dm:.4f} | p-value = {p:.6f}")

    return pd.DataFrame({
        "true": true_vals,
        "har": preds_har,
        "har_ewt": preds_ewt,
        "stack": preds_stack
    })


# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":

    _, returns = load_data(CSV_PATH)

    results = run_pipeline(returns)

    results.to_csv("btc_vol_forecast_ewt.csv")

    logging.info("Pipeline finished successfully.")
