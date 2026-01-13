from src.data_ingestion.BTC_loader import get_bitcoin_data
import numpy as np
import pandas as pd
import pywt

from arch import arch_model
from sklearn.linear_model import LinearRegression


# --------------------------------------------------
# Wavelet feature: causal rolling high-freq energy
# --------------------------------------------------
def rolling_wavelet_energy(returns, window=64, wavelet="db4"):
    energy = np.full(len(returns), np.nan)
    for t in range(window, len(returns)):
        slice_ = returns[t - window : t]
        coeffs = pywt.wavedec(slice_, wavelet, level=1)
        detail = coeffs[1]
        energy[t] = np.mean(detail ** 2)
    return energy


def main(download_data: bool = False):

    # -------------------------
    # Load data
    # -------------------------
    if download_data:
        df = get_bitcoin_data(timeframe="daily", days=3000)
        df.to_csv("data/BTC_data.csv")
    else:
        df = pd.read_csv(
            "data/BTC_data.csv",
            index_col="timestamp",
            parse_dates=True
        )

    # -------------------------
    # Log returns (scaled)
    # -------------------------
    r = np.log(df["close"]).diff().dropna().values * 100

    # -------------------------
    # BASELINE: GARCH(1,1)
    # -------------------------
    garch_base = arch_model(
        r,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal"
    )

    res_base = garch_base.fit(disp="off")

    print("\n===== BASELINE GARCH =====")
    print(res_base.summary())

    # -------------------------
    # Wavelet feature
    # -------------------------
    W = rolling_wavelet_energy(r)
    valid = ~np.isnan(W)

    r_w = r[valid]
    W = W[valid]

    # -------------------------
    # Wavelet → volatility proxy
    # -------------------------
    y = r_w ** 2

    reg = LinearRegression().fit(W.reshape(-1, 1), y)
    y_hat = reg.predict(W.reshape(-1, 1))

    resid_vol = y - y_hat
    resid_vol = np.clip(resid_vol, 1e-8, None)

    # -------------------------
    # Wavelet + GARCH
    # -------------------------
    garch_wavelet = arch_model(
        resid_vol,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal"
    )

    res_wavelet = garch_wavelet.fit(disp="off")

    print("\n===== WAVELET + GARCH =====")
    print(res_wavelet.summary())


if __name__ == "__main__":
    main()
