import numpy as np
import pandas as pd
import logging
import time
from numpy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import norm
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.decomposition import PCA
import warnings


# ==================================================
# Logging config
# ==================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)


# ==================================================
# Smooth transition function
# ==================================================
def beta(x):
    x = np.clip(x, 0.0, 1.0)
    return x**4 * (35 - 84*x + 70*x**2 - 20*x**3)


# ==================================================
# EWT boundary detection
# ==================================================
def detect_boundaries(signal, max_modes=6):

    N = len(signal)

    spectrum = np.abs(fft(signal))[: N // 2]
    spectrum[0] = 0.0

    peaks, _ = find_peaks(
        spectrum,
        distance=max(1, (N // 2) // max_modes)
    )

    if len(peaks) < 2:
        raise RuntimeError("Not enough spectral peaks")

    peaks = peaks[np.argsort(spectrum[peaks])[::-1]]
    peaks = np.sort(peaks[:max_modes])

    boundaries = 0.5 * (peaks[:-1] + peaks[1:])

    return boundaries / N


# ==================================================
# EWT decomposition
# ==================================================
def ewt_decompose(signal, max_modes=6, gamma=0.25):

    N = len(signal)

    freqs = fftfreq(N)
    abs_freqs = np.abs(freqs)

    fft_signal = fft(signal)

    boundaries = detect_boundaries(signal, max_modes)

    filters = []

    w1 = boundaries[0]

    phi = np.zeros(N)

    mask1 = abs_freqs <= (1 - gamma) * w1
    mask2 = ((1 - gamma) * w1 < abs_freqs) & (abs_freqs <= (1 + gamma) * w1)

    phi[mask1] = 1.0
    phi[mask2] = np.cos(
        np.pi / 2 * beta(
            (abs_freqs[mask2] - (1 - gamma) * w1) / (2 * gamma * w1)
        )
    )

    filters.append(phi)

    for wn, wnp1 in zip(boundaries[:-1], boundaries[1:]):

        psi = np.zeros(N)

        band = ((1 + gamma) * wn <= abs_freqs) & (abs_freqs <= (1 - gamma) * wnp1)
        up = ((1 - gamma) * wnp1 <= abs_freqs) & (abs_freqs <= (1 + gamma) * wnp1)
        down = ((1 - gamma) * wn <= abs_freqs) & (abs_freqs <= (1 + gamma) * wn)

        psi[band] = 1.0

        psi[up] = np.cos(
            np.pi / 2 * beta(
                (abs_freqs[up] - (1 - gamma) * wnp1) / (2 * gamma * wnp1)
            )
        )

        psi[down] = np.sin(
            np.pi / 2 * beta(
                (abs_freqs[down] - (1 - gamma) * wn) / (2 * gamma * wn)
            )
        )

        filters.append(psi)

    components = [
        np.real(ifft(fft_signal * filt))
        for filt in filters
    ]

    return components


# ==================================================
# GARCH
# ==================================================
def fit_garch(series):

    series = np.asarray(series)
    series = series[~np.isnan(series)]

    if len(series) < 50:
        raise RuntimeError("Series too short")

    model = arch_model(
        series,
        mean="constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=False
    )

    return model.fit(disp="off")


# ==================================================
# EWT-PCA-GARCH Implementation
# ==================================================
def ewt_pca_garch_pipeline(train_series, n_components=None, variance_threshold=0.95):
    """
    Hybrid EWT+PCA+GARCH pipeline
    
    Parameters:
    -----------
    train_series : array-like
        Training time series
    n_components : int or None
        Number of PCA components to keep (None for variance threshold)
    variance_threshold : float
        Minimum variance to explain if n_components is None
    
    Returns:
    --------
    tuple : (volatility_forecast, pca_model, selected_pcs, pca_components)
    """
    try:
        # Step 1: EWT Decomposition
        ewt_components = ewt_decompose(train_series, max_modes=8)
        
        # Convert to numpy array and handle potential NaN values
        components_matrix = np.array(ewt_components).T  # Shape: (n_samples, n_components)
        
        # Clean NaN values
        nan_mask = np.any(np.isnan(components_matrix), axis=1)
        if np.any(nan_mask):
            logger.warning(f"Removing {np.sum(nan_mask)} NaN rows from EWT components")
            components_matrix = components_matrix[~nan_mask]
        
        # Standardize components (important for PCA)
        component_means = np.mean(components_matrix, axis=0)
        component_stds = np.std(components_matrix, axis=0)
        components_standardized = (components_matrix - component_means) / np.where(component_stds > 0, component_stds, 1)
        
        # Step 2: PCA on EWT components
        pca = PCA(n_components=n_components)
        pca.fit(components_standardized)
        
        # Determine number of components to keep
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        if n_components is None:
            n_keep = np.argmax(explained_variance >= variance_threshold) + 1
            n_keep = min(n_keep, len(ewt_components))
        else:
            n_keep = min(n_components, len(ewt_components))
        
        # Transform to PCA space
        pcs = pca.transform(components_standardized)
        selected_pcs = pcs[:, :n_keep]
        
        logger.info(f"EWT decomposed into {len(ewt_components)} components")
        logger.info(f"PCA selected {n_keep} components explaining {explained_variance[n_keep-1]:.2%} of variance")
        
        # Step 3: GARCH modeling on principal components
        
        # Option A: Fit GARCH to each significant PC
        pc_vol_forecasts = []
        pc_models = []
        
        for i in range(n_keep):
            pc_series = selected_pcs[:, i]
            
            # Skip if series has no variation
            if np.std(pc_series) < 1e-10:
                continue
                
            try:
                # Fit GARCH to each PC
                garch_model = fit_garch(pc_series)
                
                # Forecast volatility
                pc_vol = np.sqrt(
                    garch_model.forecast(horizon=1).variance.values[-1, 0]
                )
                
                # Weight by variance explained
                weight = pca.explained_variance_ratio_[i]
                weighted_vol = pc_vol * np.sqrt(weight)
                
                pc_vol_forecasts.append(weighted_vol)
                pc_models.append(garch_model)
                
            except Exception as e:
                logger.warning(f"GARCH failed for PC {i}: {e}")
                continue
        
        if not pc_vol_forecasts:
            raise RuntimeError("No valid PC GARCH models")
        
        # Aggregate volatility forecasts (Euclidean norm of weighted volatilities)
        total_vol_forecast = np.sqrt(np.sum(np.square(pc_vol_forecasts)))
        
        # Alternative: Reconstruct and fit GARCH on residuals
        try:
            # Reconstruct signal from selected PCs
            reconstructed = pca.inverse_transform(
                np.hstack([
                    selected_pcs,
                    np.zeros((selected_pcs.shape[0], pca.n_components_ - n_keep))
                ])
            )
            
            # Denormalize
            reconstructed = reconstructed * component_stds + component_means
            
            # Calculate residuals (approximation error)
            residuals = train_series[-len(reconstructed):] - np.sum(reconstructed, axis=1)
            
            # Fit GARCH to residuals if they're meaningful
            if np.std(residuals) > 0.1 * np.std(train_series):
                res_garch = fit_garch(residuals)
                res_vol = np.sqrt(
                    res_garch.forecast(horizon=1).variance.values[-1, 0]
                )
                # Combine with PC volatilities
                total_vol_forecast = np.sqrt(total_vol_forecast**2 + res_vol**2)
        
        except Exception as e:
            logger.warning(f"Residual GARCH failed: {e}")
        
        return total_vol_forecast, pca, selected_pcs, ewt_components
        
    except Exception as e:
        logger.error(f"EWT-PCA-GARCH pipeline failed: {e}")
        raise


# ==================================================
# Rolling OOS
# ==================================================
def rolling_oos_eval(
    returns,
    realized_vol,
    window,
    max_modes=6,
    gamma=0.25
):

    n_total = len(returns)
    n_steps = n_total - window
    logger.info(f"OOS steps: {n_steps}")

    garch_plain = np.full(n_steps, np.nan)
    garch_ewt = np.full(n_steps, np.nan)
    arima_low_garch_high = np.full(n_steps, np.nan)
    svm_ewt = np.full(n_steps, np.nan)
    ewt_pca_garch = np.full(n_steps, np.nan)  # New method
    rv_true = np.full(n_steps, np.nan)

    t0 = time.time()

    for i, t in enumerate(range(window, n_total)):

        train = returns[t - window: t]

        # ---------- Plain GARCH ----------
        try:
            res_g = fit_garch(train)
            garch_plain[i] = np.sqrt(
                res_g.forecast(horizon=1).variance.values[-1, 0]
            )
        except Exception as e:
            logger.warning(f"Plain GARCH failed at t={t}: {e}")

        # ---------- EWT pipeline ----------
        try:
            components = ewt_decompose(train, max_modes=max_modes, gamma=gamma)

            # ---- EWT-GARCH ----
            vols = []
            for comp in components:
                if np.std(comp) < 1e-10:
                    continue
                res_c = fit_garch(comp)
                v = np.sqrt(res_c.forecast(horizon=1).variance.values[-1, 0])
                vols.append(v)

            if vols:
                garch_ewt[i] = np.sqrt(np.sum(np.square(vols)))

            # ---- ARIMA low + GARCH high ----
            low_freq = components[0]
            high_freqs = components[1:]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                arima = ARIMA(low_freq, order=(1, 0, 1),
                              enforce_stationarity=False,
                              enforce_invertibility=False).fit()
                low_forecast = np.abs(arima.forecast()[0])

            vols = []
            for comp in high_freqs:
                if np.std(comp) < 1e-10:
                    continue
                res_c = fit_garch(comp)
                v = np.sqrt(res_c.forecast(horizon=1).variance.values[-1, 0])
                vols.append(v)

            hf_vol = np.sqrt(np.sum(np.square(vols))) if vols else 0.0
            arima_low_garch_high[i] = low_forecast + hf_vol

            # ---- SVM EWT ----
            vols = []
            for comp in components:
                if np.std(comp) < 1e-10:
                    continue

                comp_var = comp**2

                if len(comp_var) < 5:
                    continue

                X = comp_var[:-1].reshape(-1, 1)
                y = comp_var[1:]

                model = SVR(kernel="rbf", C=1.0, epsilon=1e-6)
                model.fit(X, y)

                pred_var = model.predict(comp_var[-1].reshape(1, -1))[0]
                vols.append(max(pred_var, 0))

            if vols:
                svm_ewt[i] = np.sqrt(np.sum(vols))

        except Exception as e:
            logger.warning(f"EWT pipeline failed at t={t}: {e}")

        # ---------- NEW: EWT-PCA-GARCH ----------
        try:
            # You can adjust parameters here:
            # Option 1: Fixed number of components
            # vol_forecast, _, _, _ = ewt_pca_garch_pipeline(train, n_components=3)
            
            # Option 2: Variance threshold (recommended)
            vol_forecast, _, _, _ = ewt_pca_garch_pipeline(
                train, 
                n_components=None, 
                variance_threshold=0.95
            )
            
            ewt_pca_garch[i] = vol_forecast
            logger.debug(f"EWT-PCA-GARCH forecast at t={t}: {vol_forecast:.6f}")
            
        except Exception as e:
            logger.warning(f"EWT-PCA-GARCH failed at t={t}: {e}")

        rv_true[i] = realized_vol[t]

        if i % 100 == 0 and i > 0:
            speed = i / (time.time() - t0)
            logger.info(f"{i}/{n_steps} | {speed:.2f} steps/sec")

    mask = (
        ~np.isnan(garch_plain) &
        ~np.isnan(garch_ewt) &
        ~np.isnan(arima_low_garch_high) &
        ~np.isnan(svm_ewt) &
        ~np.isnan(ewt_pca_garch)  # Include new method in mask
    )

    logger.info(f"Valid forecasts: {np.sum(mask)}")
    logger.info(f"EWT-PCA-GARCH success rate: {np.sum(~np.isnan(ewt_pca_garch))}/{n_steps}")

    return {
        "rv": rv_true[mask],
        "garch_plain": garch_plain[mask],
        "garch_ewt": garch_ewt[mask],
        "arima_low_garch_high": arima_low_garch_high[mask],
        "svm_ewt": svm_ewt[mask],
        "ewt_pca_garch": ewt_pca_garch[mask],  # Add new method to results
    }


# ==================================================
# Diebold–Mariano
# ==================================================
def diebold_mariano(y, f1, f2, loss="mse"):

    if loss == "mse":
        d = (y - f1)**2 - (y - f2)**2
    else:
        d = np.abs(y - f1) - np.abs(y - f2)

    DM = np.mean(d) / np.sqrt(np.var(d) / len(d))
    p = 2 * (1 - norm.cdf(abs(DM)))

    return DM, p


# ==================================================
# Visualization Helper (Optional)
# ==================================================
def plot_ewt_pca_components(pca_model, selected_pcs, ewt_components, title="EWT-PCA Analysis"):
    """
    Visualize the EWT-PCA decomposition
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Plot 1: Explained variance
        ax = axes[0, 0]
        explained_var = np.cumsum(pca_model.explained_variance_ratio_)
        ax.bar(range(1, len(explained_var) + 1), pca_model.explained_variance_ratio_, 
               alpha=0.6, label='Individual')
        ax.step(range(1, len(explained_var) + 1), explained_var, 
                where='mid', label='Cumulative', color='red')
        ax.axhline(y=0.95, color='green', linestyle='--', label='95% threshold')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('PCA Explained Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: First few EWT components
        ax = axes[0, 1]
        n_plot = min(4, len(ewt_components))
        for j in range(n_plot):
            ax.plot(ewt_components[j][-200:], label=f'Mode {j+1}', alpha=0.7)
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.set_title('First 4 EWT Components (last 200 points)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: First few principal components
        ax = axes[1, 0]
        n_pcs_plot = min(3, selected_pcs.shape[1])
        for j in range(n_pcs_plot):
            ax.plot(selected_pcs[-200:, j], label=f'PC {j+1}', alpha=0.7)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('First 3 Principal Components (last 200 points)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Scree plot alternative
        ax = axes[1, 1]
        ax.plot(pca_model.explained_variance_, 'o-')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Scree Plot')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: PCA loadings (first PC)
        ax = axes[2, 0]
        loadings = pca_model.components_[:n_plot, :n_plot]
        im = ax.imshow(loadings, cmap='RdBu_r', aspect='auto', 
                      vmin=-np.max(np.abs(loadings)), vmax=np.max(np.abs(loadings)))
        ax.set_xlabel('EWT Component')
        ax.set_ylabel('Principal Component')
        ax.set_title('PCA Loadings Matrix (subset)')
        plt.colorbar(im, ax=ax)
        
        # Plot 6: Volatility comparison
        ax = axes[2, 1]
        # This would be filled during the main execution
        ax.text(0.5, 0.5, 'Volatility forecasts\nwill be shown during\nexecution', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Volatility Forecasts')
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        logger.warning("Matplotlib not available for visualization")


# ==================================================
# Main
# ==================================================
if __name__ == "__main__":

    df = pd.read_csv(
        r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv"
    )

    df = df[df["timestamp"] >= "2023-04-01"]

    df["ret"] = np.log(df["close"]).diff()
    df["rv"] = np.abs(df["ret"])

    df.dropna(inplace=True)

    window = 1000

    results = rolling_oos_eval(
        returns=df["ret"].values,
        realized_vol=df["rv"].values,
        window=window,
        max_modes=10,
        gamma=0.2
    )

    logger.info("===== METRICS =====")

    for k in results.keys():
        if k == "rv":
            continue
        mse = mean_squared_error(results["rv"], results[k]) * 100
        mae = mean_absolute_error(results["rv"], results[k]) * 100
        logger.info(f"{k.upper():<25} | MSE={mse:.6f} | MAE={mae:.6f}")

    logger.info("===== DM TESTS =====")
    
    # Compare all methods against plain GARCH
    DM1, p1 = diebold_mariano(
        results["rv"],
        results["garch_plain"],
        results["garch_ewt"]
    )
    logger.info(f"GARCH vs EWT              : DM={DM1:.4f} p={p1:.6f}")

    DM2, p2 = diebold_mariano(
        results["rv"],
        results["garch_plain"],
        results["arima_low_garch_high"]
    )
    logger.info(f"GARCH vs ARIMA+EWT        : DM={DM2:.4f} p={p2:.6f}")

    DM3, p3 = diebold_mariano(
        results["rv"],
        results["garch_plain"],
        results["svm_ewt"]
    )
    logger.info(f"GARCH vs SVM+EWT          : DM={DM3:.4f} p={p3:.6f}")

    DM4, p4 = diebold_mariano(
        results["rv"],
        results["garch_plain"],
        results["ewt_pca_garch"]
    )
    logger.info(f"GARCH vs EWT-PCA-GARCH    : DM={DM4:.4f} p={p4:.6f}")

    # Compare EWT-PCA-GARCH against other EWT methods
    DM5, p5 = diebold_mariano(
        results["rv"],
        results["garch_ewt"],
        results["ewt_pca_garch"]
    )
    logger.info(f"EWT-GARCH vs EWT-PCA-GARCH: DM={DM5:.4f} p={p5:.6f}")