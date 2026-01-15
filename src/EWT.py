import numpy as np
from numpy.fft import fft, ifft, fftfreq
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --------------------------------------------------
# Smooth transition function beta(x)
# --------------------------------------------------
def beta(x):
    x = np.clip(x, 0, 1)
    return x**4 * (35 - 84*x + 70*x**2 - 20*x**3)

# --------------------------------------------------
# Boundary detection (Algorithm 1)
# --------------------------------------------------
def detect_boundaries(signal, alpha=0.3):
    N = len(signal)
    spectrum = np.abs(fft(signal))
    spectrum = spectrum[:N//2]

    spec_norm = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())
    threshold = spec_norm.min() + alpha * (spec_norm.max() - spec_norm.min())

    idx = np.where(spec_norm >= threshold)[0]
    if len(idx) < 2:
        raise ValueError("Not enough spectral peaks detected")

    boundaries = [(idx[i] + idx[i+1]) / 2 for i in range(len(idx)-1)]
    return np.array(boundaries)

# --------------------------------------------------
# Empirical filter bank
# --------------------------------------------------
def build_ewt_filters(freqs, boundaries, gamma=0.25):
    filters = []

    # Scaling function
    w1 = boundaries[0]
    phi = np.zeros_like(freqs)

    for i, w in enumerate(np.abs(freqs)):
        if w <= (1 - gamma) * w1:
            phi[i] = 1
        elif (1 - gamma) * w1 < w <= (1 + gamma) * w1:
            phi[i] = np.cos(
                np.pi/2 * beta((w - (1 - gamma)*w1) / (2*gamma*w1))
            )

    filters.append(phi)

    # Wavelets
    for n in range(len(boundaries) - 1):
        wn, wnp1 = boundaries[n], boundaries[n+1]
        psi = np.zeros_like(freqs)

        for i, w in enumerate(np.abs(freqs)):
            if (1 + gamma)*wn <= w <= (1 - gamma)*wnp1:
                psi[i] = 1
            elif (1 - gamma)*wnp1 <= w <= (1 + gamma)*wnp1:
                psi[i] = np.cos(
                    np.pi/2 * beta((w - (1 - gamma)*wnp1) / (2*gamma*wnp1))
                )
            elif (1 - gamma)*wn <= w <= (1 + gamma)*wn:
                psi[i] = np.sin(
                    np.pi/2 * beta((w - (1 - gamma)*wn) / (2*gamma*wn))
                )

        filters.append(psi)

    return filters

# --------------------------------------------------
# EWT decomposition (Eq. 5 & 6)
# --------------------------------------------------
def ewt_decompose(signal, alpha=0.3, gamma=0.25):
    N = len(signal)
    freqs = fftfreq(N)
    fft_signal = fft(signal)

    boundaries = detect_boundaries(signal, alpha)
    filters = build_ewt_filters(freqs, boundaries, gamma)

    components = []
    for filt in filters:
        comp = np.real(ifft(fft_signal * filt))
        components.append(comp)

    return components

# --------------------------------------------------
# Reconstruction (Eq. 7–9)
# --------------------------------------------------
def ewt_reconstruct(components):
    return np.sum(components, axis=0)

from arch import arch_model
import numpy as np

def fit_garch(
    series,
    p=1,
    q=1,
    mean="Zero",
    dist="t",
    rescale=True
):
    """
    Fits a GARCH(p, q) model to a return series.

    Parameters
    ----------
    series : array-like
        Return series (must be stationary).
    p, q : int
        GARCH orders.
    mean : str
        'Zero' or 'Constant'.
    dist : str
        'normal' or 't'.
    rescale : bool
        Rescales data internally for numerical stability.

    Returns
    -------
    result : arch.univariate.base.ARCHModelResult
        Fitted GARCH model.
    cond_vol : np.ndarray
        Conditional volatility.
    """

    series = np.asarray(series)
    series = series[~np.isnan(series)]

    if len(series) < 50:
        raise ValueError("Series too short for GARCH estimation")

    model = arch_model(
        series,
        mean=mean,
        vol="GARCH",
        p=p,
        q=q,
        dist=dist,
        rescale=rescale
    )

    result = model.fit(disp="off")

    cond_vol = result.conditional_volatility

    return result, cond_vol

from sklearn.metrics import mean_squared_error, mean_absolute_error

# --------------------------------------------------
# Rolling OOS evaluation
# --------------------------------------------------
def rolling_oos_eval(
    returns,
    realized_vol,
    window=500,
    alpha=0.01,
    gamma=0.01
):
    garch_preds = []
    ewt_garch_preds = []
    rv_true = []

    for t in range(window, len(returns)):
        train = returns[t-window:t]

        # ---------------------------
        # Plain GARCH
        # ---------------------------
        try:
            res_g, _ = fit_garch(train)
            garch_forecast = np.sqrt(
                res_g.forecast(horizon=1).variance.values[-1, 0]
            )
        except:
            garch_forecast = np.nan

        # ---------------------------
        # EWT → GARCH
        # ---------------------------
        try:
            components = ewt_decompose(train, alpha=alpha, gamma=gamma)
            vols = []

            for comp in components:
                if np.std(comp) == 0:
                    continue

                res_c, _ = fit_garch(comp)
                v = np.sqrt(
                    res_c.forecast(horizon=1).variance.values[-1, 0]
                )
                vols.append(v)

            ewt_forecast = np.sum(vols) if len(vols) > 0 else np.nan

        except:
            ewt_forecast = np.nan

        garch_preds.append(garch_forecast)
        ewt_garch_preds.append(ewt_forecast)
        rv_true.append(realized_vol[t])

    garch_preds = np.array(garch_preds)
    ewt_garch_preds = np.array(ewt_garch_preds)
    rv_true = np.array(rv_true)

    mask = ~np.isnan(garch_preds) & ~np.isnan(ewt_garch_preds)

    return {
        "GARCH_MSE": mean_squared_error(rv_true[mask], garch_preds[mask]),
        "EWT_GARCH_MSE": mean_squared_error(rv_true[mask], ewt_garch_preds[mask]),
        "GARCH_MAE": mean_absolute_error(rv_true[mask], garch_preds[mask]),
        "EWT_GARCH_MAE": mean_absolute_error(rv_true[mask], ewt_garch_preds[mask]),
    }



if __name__ == "__main__":
    df = pd.read_csv(r'C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv')

    df['ret'] = np.log(df['close']).diff() * 100
    df['rv'] = df['ret'].abs()  # realized volatility proxy
    df = df.dropna()

    results = rolling_oos_eval(
        returns=df['ret'].values,
        realized_vol=df['rv'].values,
        window=500,
        alpha=0.01,
        gamma=0.01
    )

    print("\n===== ROLLING OOS RESULTS =====")
    for k, v in results.items():
        print(f"{k}: {v:.6f}")
