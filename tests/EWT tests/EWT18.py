import numpy as np
import pandas as pd
import logging
from arch import arch_model
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import norm

# ======================================================
# CONFIG
# ======================================================

CSV_PATH = "data.csv"   # must contain: timestamp, close
WINDOW_TRAIN = 2000
RV_WINDOW = 5           # realized vol from intraday proxy
EPS = 1e-8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ======================================================
# DATA LOADING
# ======================================================

def load_data(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")

    price = df["close"]

    returns = np.log(price).diff().dropna()

    return price, returns


# ======================================================
# REALIZED VOLATILITY (PROXY)
# ======================================================

def realized_volatility(returns, window):
    rv = returns.rolling(window).apply(lambda x: np.sqrt(np.sum(x**2)))
    return rv.dropna()


# ======================================================
# HAR FEATURES
# ======================================================

def har_features(rv):

    df = pd.DataFrame({"rv": rv})

    df["d"] = df["rv"].shift(1)
    df["w"] = df["rv"].rolling(5).mean().shift(1)
    df["m"] = df["rv"].rolling(22).mean().shift(1)

    df = df.dropna()

    return df


# ======================================================
# HAR MODEL (LOG VOL)
# ======================================================

def fit_har(train):

    X = train[["d", "w", "m"]].values
    y = np.log(train["rv"].values + EPS)

    X = np.column_stack([np.ones(len(X)), X])

    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    return beta


def predict_har(beta, row):

    x = np.array([1, row["d"], row["w"], row["m"]])
    y_hat = x @ beta

    return np.exp(y_hat)


# ======================================================
# GARCH
# ======================================================

def fit_garch(returns):

    model = arch_model(
        returns * 100,
        vol="Garch",
        p=1,
        q=1,
        dist="normal"
    )

    res = model.fit(disp="off")

    return res


def predict_garch(res):

    f = res.forecast(horizon=1)
    sigma = np.sqrt(f.variance.values[-1, 0])

    return sigma / 100


# ======================================================
# DIEBOLD-MARIANO (MAE LOSS)
# ======================================================

def diebold_mariano(e1, e2):

    d = e1 - e2

    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)

    dm = mean_d / np.sqrt(var_d / len(d))
    p = 2 * (1 - norm.cdf(abs(dm)))

    return dm, p


# ======================================================
# MAIN PIPELINE
# ======================================================

def run_pipeline(price, returns):

    rv = realized_volatility(returns, RV_WINDOW)
    har_df = har_features(rv)

    returns = returns.loc[har_df.index]

    preds_har = []
    preds_garch = []
    preds_stack = []
    true_vals = []

    stack_X = []
    stack_y = []

    logging.info(f"Train window: {WINDOW_TRAIN}")
    logging.info(f"Test window : {len(har_df) - WINDOW_TRAIN}")

    for t in range(WINDOW_TRAIN, len(har_df)):

        train = har_df.iloc[t - WINDOW_TRAIN:t]
        test = har_df.iloc[t]

        # ------------------
        # HAR backbone
        # ------------------

        beta = fit_har(train)
        har_pred = predict_har(beta, test)

        # ------------------
        # GARCH
        # ------------------

        ret_train = returns.iloc[t - WINDOW_TRAIN:t]
        garch_fit = fit_garch(ret_train)
        garch_pred = predict_garch(garch_fit)

        # ------------------
        # Residual target
        # ------------------

        residual = test["rv"] - har_pred

        stack_X.append([har_pred, garch_pred])
        stack_y.append(residual)

        preds_har.append(har_pred)
        preds_garch.append(garch_pred)
        true_vals.append(test["rv"])

    stack_X = np.array(stack_X)
    stack_y = np.array(stack_y)
    true_vals = np.array(true_vals)
    preds_har = np.array(preds_har)
    preds_garch = np.array(preds_garch)

    # ======================================================
    # META LEARNER — MAE OPTIMAL
    # ======================================================

    meta = QuantileRegressor(
        quantile=0.5,
        alpha=0.0,
        solver="highs"
    )

    meta.fit(stack_X, stack_y)

    correction = meta.predict(stack_X)

    preds_stack = preds_har + correction

    # ======================================================
    # METRICS
    # ======================================================

    har_mse = mean_squared_error(true_vals, preds_har)
    har_mae = mean_absolute_error(true_vals, preds_har)

    garch_mse = mean_squared_error(true_vals, preds_garch)
    garch_mae = mean_absolute_error(true_vals, preds_garch)

    stack_mse = mean_squared_error(true_vals, preds_stack)
    stack_mae = mean_absolute_error(true_vals, preds_stack)

    logging.info("====== OOS RESULTS (volatility scale) ======")
    logging.info(f"HAR    | MSE={har_mse:.6f} | MAE={har_mae:.6f}")
    logging.info(f"GARCH  | MSE={garch_mse:.6f} | MAE={garch_mae:.6f}")
    logging.info(f"STACK  | MSE={stack_mse:.6f} | MAE={stack_mae:.6f}")

    # ======================================================
    # DIEBOLD-MARIANO
    # ======================================================

    e_har = np.abs(true_vals - preds_har)
    e_stack = np.abs(true_vals - preds_stack)

    dm, p = diebold_mariano(e_stack, e_har)

    logging.info(f"DM (STACK vs HAR) = {dm:.4f} | p-value = {p:.6f}")

    return pd.DataFrame({
        "true": true_vals,
        "har": preds_har,
        "garch": preds_garch,
        "stack": preds_stack
    })


# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":

    price, returns = load_data(CSV_PATH)

    results = run_pipeline(price, returns)

    results.to_csv("forecast_results.csv")

    logging.info("Pipeline finished successfully.")
