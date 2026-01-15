import numpy as np
import pandas as pd
from numpy.fft import fft, ifft, fftfreq
from sklearn.metrics import mean_squared_error, mean_absolute_error
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# GARCH helper
# ======================================================
def garch_forecast(train_returns, p=1, q=1):
    model = arch_model(
        train_returns,
        mean="Zero",
        vol="GARCH",
        p=p,
        q=q,
        dist="t",
        rescale=True
    )
    res = model.fit(disp="off")
    fcast = res.forecast(horizon=1, reindex=False)
    sigma = np.sqrt(fcast.variance.values[-1, 0])
    return sigma

# ======================================================
# VERY BASIC EWT (keep your own if you want)
# This is only to show evaluation correctness
# ======================================================
def simple_ewt(signal, K=3):
    """
    Extremely simplified spectral split
    (not theoretically perfect, but causal-safe here)
    """
    N = len(signal)
    X = fft(signal)
    freqs = fftfreq(N)

    bands = np.linspace(0, 0.5, K + 1)
    components = []

    for k in range(K):
        mask = (np.abs(freqs) >= bands[k]) & (np.abs(freqs) < bands[k + 1])
        comp = np.real(ifft(X * mask))
        components.append(comp)

    return components

# ======================================================
# Rolling OOS evaluation
# ======================================================
def rolling_oos_evaluation(returns, rv, start=500):
    garch_preds = []
    ewt_garch_preds = []

    for t in range(start, len(returns)):
        train = returns[:t]

        # ---------- Plain GARCH ----------
        sigma_garch = garch_forecast(train)
        garch_preds.append(sigma_garch)

        # ---------- EWT → GARCH ----------
        comps = simple_ewt(train, K=3)
        sigmas = []

        for c in comps:
            if np.std(c) < 1e-8:
                continue
            sigmas.append(garch_forecast(c))

        sigma_ewt = np.sqrt(np.sum(np.array(sigmas) ** 2))
        ewt_garch_preds.append(sigma_ewt)

    rv_test = rv[start:]

    return (
        np.array(garch_preds),
        np.array(ewt_garch_preds),
        rv_test
    )


from scipy.stats import ttest_1samp



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
    rv = np.abs(returns)  # realized volatility proxy

    garch_pred, ewt_pred, rv_test = rolling_oos_evaluation(
        returns, rv, start=500
    )

    print("===== OUT-OF-SAMPLE RESULTS =====")
    print("Plain GARCH")
    print("  MSE :", mean_squared_error(rv_test, garch_pred))
    print("  MAE :", mean_absolute_error(rv_test, garch_pred))

    print("\nEWT → GARCH")
    print("  MSE :", mean_squared_error(rv_test, ewt_pred))
    print("  MAE :", mean_absolute_error(rv_test, ewt_pred))

    d = np.abs(rv_test - garch_pred) - np.abs(rv_test - ewt_pred)
    t_stat, p_value = ttest_1samp(d, 0)

    print("DM-style test")
    print("t-stat :", t_stat)
    print("p-val  :", p_value)