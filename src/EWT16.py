import numpy as np
import pandas as pd
import logging
import warnings
import traceback
from typing import Tuple, Dict, List

from numpy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks
from scipy.optimize import minimize
from arch import arch_model
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ======================================================
# Logging Configuration
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger()

# ======================================================
# EWT Functions (Optimized and Robust)
# ======================================================
def smooth_beta(x: np.ndarray) -> np.ndarray:
    """Smooth transition function."""
    x = np.clip(x, 0.0, 1.0)
    return x**4 * (35 - 84*x + 70*x**2 - 20*x**3)

def detect_boundaries_returns(returns: np.ndarray, max_modes: int = 5) -> np.ndarray:
    """
    Detect EWT boundaries on RETURN series (not RV).
    Returns normalized frequencies (0..0.5).
    """
    N = len(returns)
    if N < 50:  # Increased minimum for stable frequency detection
        raise ValueError(f"Signal too short ({N}) for boundary detection")
    
    # Remove mean to focus on oscillations
    signal = returns - np.mean(returns)
    spectrum = np.abs(fft(signal))[: N // 2]
    spectrum[0] = 0.0  # Remove DC component
    
    # Find peaks with proper distance constraint
    min_distance = max(1, (N // 2) // (max_modes * 4))
    peaks, properties = find_peaks(spectrum, distance=min_distance, prominence=0.1*np.max(spectrum))
    
    if len(peaks) < 2:
        # Fallback to equally spaced boundaries
        log.warning("Not enough spectral peaks, using equally spaced boundaries")
        boundaries = np.linspace(0.1, 0.4, max_modes-1)
        return boundaries
    
    # Select top peaks by prominence
    if 'prominences' in properties:
        prominences = properties['prominences']
        top_idx = np.argsort(prominences)[-min(max_modes, len(prominences)):]
        peaks = peaks[np.sort(top_idx)]
    
    # Convert to normalized frequencies and create boundaries at midpoints
    freqs = peaks / float(N)
    freqs = np.sort(freqs)
    boundaries = 0.5 * (freqs[:-1] + freqs[1:])
    
    # Ensure boundaries are within valid range
    boundaries = np.clip(boundaries, 0.05, 0.45)
    log.info(f"Detected {len(boundaries)} boundaries at {boundaries}")
    return boundaries

def ewt_decompose_returns(returns: np.ndarray, boundaries: np.ndarray, 
                         gamma: float = 0.1) -> List[np.ndarray]:
    """
    Apply EWT to return series.
    Returns orthogonal components that sum to original returns.
    """
    N = len(returns)
    signal = returns - np.mean(returns)  # Work with zero-mean series
    
    # FFT of signal
    freqs = np.abs(fftfreq(N))
    fft_sig = fft(signal)
    
    # Sort and validate boundaries
    b = np.sort(np.asarray(boundaries))
    b = np.clip(b, 1e-8, 0.5 - 1e-8)
    num_bands = len(b) + 1
    
    filters = []
    
    # Low-pass filter (first band)
    w1 = b[0]
    phi = np.zeros(N, dtype=complex)
    # Flat region
    flat_low = freqs <= (1 - gamma) * w1
    phi[flat_low] = 1.0
    # Transition region
    trans_low = ((1 - gamma) * w1 < freqs) & (freqs <= (1 + gamma) * w1)
    if np.any(trans_low):
        beta_vals = (freqs[trans_low] - (1 - gamma) * w1) / (2 * gamma * w1)
        phi[trans_low] = np.cos(0.5 * np.pi * smooth_beta(beta_vals))
    filters.append(phi)
    
    # Band-pass filters (middle bands)
    for i in range(len(b) - 1):
        w0 = b[i]
        w1 = b[i + 1]
        psi = np.zeros(N, dtype=complex)
        
        # Lower transition
        trans_lower = ((1 - gamma) * w0 <= freqs) & (freqs < (1 + gamma) * w0)
        if np.any(trans_lower):
            beta_lower = (freqs[trans_lower] - (1 - gamma) * w0) / (2 * gamma * w0)
            psi[trans_lower] = np.sin(0.5 * np.pi * smooth_beta(beta_lower))
        
        # Flat region
        flat_band = ((1 + gamma) * w0 <= freqs) & (freqs <= (1 - gamma) * w1)
        psi[flat_band] = 1.0
        
        # Upper transition
        trans_upper = ((1 - gamma) * w1 < freqs) & (freqs <= (1 + gamma) * w1)
        if np.any(trans_upper):
            beta_upper = (freqs[trans_upper] - (1 - gamma) * w1) / (2 * gamma * w1)
            psi[trans_upper] = np.cos(0.5 * np.pi * smooth_beta(beta_upper))
        
        filters.append(psi)
    
    # High-pass filter (last band)
    w_last = b[-1]
    psi_high = np.zeros(N, dtype=complex)
    # Transition region
    trans_high = ((1 - gamma) * w_last <= freqs) & (freqs <= (1 + gamma) * w_last)
    if np.any(trans_high):
        beta_high = (freqs[trans_high] - (1 - gamma) * w_last) / (2 * gamma * w_last)
        psi_high[trans_high] = np.sin(0.5 * np.pi * smooth_beta(beta_high))
    # Flat region
    flat_high = freqs > (1 + gamma) * w_last
    psi_high[flat_high] = 1.0
    filters.append(psi_high)
    
    # Apply filters and reconstruct components
    components = []
    for f in filters:
        comp = np.real(ifft(fft_sig * f))
        components.append(comp)
    
    # Verify reconstruction (allow small numerical error)
    reconstruction = np.sum(components, axis=0)
    mse_recon = np.mean((signal - reconstruction)**2)
    if mse_recon > 1e-10:
        log.warning(f"EWT reconstruction MSE high: {mse_recon:.2e}")
    
    log.info(f"EWT produced {len(components)} components")
    return components

# ======================================================
# Econometric Benchmarks (Proper Implementation)
# ======================================================
def realized_volatility(returns: np.ndarray, window: int = 1) -> np.ndarray:
    """Compute realized volatility (sqrt of realized variance)."""
    # For daily RV, use squared returns
    if window == 1:
        rv = returns**2
    else:
        # Rolling window RV
        rv = pd.Series(returns**2).rolling(window).sum().values
    return np.sqrt(np.maximum(rv, 1e-12))

def har_rv_forecast(rv_series: np.ndarray) -> float:
    """
    HAR-RV model: RV(t+1) = β0 + β1*RV(t) + β2*RV(t-4:t) + β3*RV(t-21:t)
    Implemented with OLS on rolling window.
    """
    try:
        n = len(rv_series)
        if n < 30:
            return max(float(rv_series[-1]), 1e-8)
        
        # Create HAR features
        df = pd.DataFrame({'rv': rv_series})
        df['rv_d'] = df['rv']  # Daily
        df['rv_w'] = df['rv'].rolling(5).mean()   # Weekly (5 days)
        df['rv_m'] = df['rv'].rolling(22).mean()  # Monthly (22 days)
        df = df.dropna()
        
        if len(df) < 22:
            return max(float(rv_series[-1]), 1e-8)
        
        # Fit HAR model
        X = df[['rv_d', 'rv_w', 'rv_m']].values[:-1]
        y = df['rv'].values[1:]
        
        # Add constant
        X = np.column_stack([np.ones(len(X)), X])
        
        # OLS with regularization for stability
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            # Forecast
            last_vals = np.array([1.0, df['rv_d'].iloc[-1], 
                                 df['rv_w'].iloc[-1], df['rv_m'].iloc[-1]])
            forecast = np.dot(last_vals, beta)
        except np.linalg.LinAlgError:
            # Fallback to simple average
            forecast = np.mean(y[-5:])
        
        return max(float(forecast), 1e-8)
    except Exception as e:
        log.warning(f"HAR-RV failed: {e}, using persistence")
        return max(float(rv_series[-1]), 1e-8)

def garch_forecast_robust(returns: np.ndarray, p: int = 1, q: int = 1) -> float:
    """
    Robust GARCH(1,1) implementation with multiple fallbacks.
    """
    try:
        # Remove mean
        ret_clean = returns - np.mean(returns)
        
        # Fit GARCH with multiple attempts
        for dist in ['normal', 't']:
            try:
                model = arch_model(ret_clean, vol='GARCH', p=p, q=q, 
                                 dist=dist, rescale=False)
                res = model.fit(disp='off', show_warning=False)
                
                # Get forecast
                forecast = res.forecast(horizon=1)
                variance = forecast.variance.values[-1, 0]
                
                # Check validity
                if variance > 0 and not np.isnan(variance):
                    return max(float(variance), 1e-8)
            except:
                continue
        
        # Fallback: EWMA variance
        alpha = 0.94
        variance = returns[-1]**2
        for r in returns[-2::-1]:
            variance = alpha * variance + (1 - alpha) * r**2
        return max(float(variance), 1e-8)
        
    except Exception as e:
        log.warning(f"All GARCH attempts failed: {e}")
        # Simple historical variance
        return max(float(np.var(returns[-22:])), 1e-8)

def persistence_forecast(rv_series: np.ndarray) -> float:
    """Simple persistence model: RV(t+1) = RV(t)."""
    return max(float(rv_series[-1]), 1e-8)

# ======================================================
# EWT-Based Volatility Model
# ======================================================
class EWTVolatilityForecaster:
    """
    EWT-based volatility model with proper variance decomposition.
    Decomposes returns, models each component, and recombines variances.
    """
    
    def __init__(self, n_components: int = 4, gamma: float = 0.1):
        self.n_components = n_components
        self.gamma = gamma
        self.boundaries = None
        self.component_models = []
        
    def fit(self, returns: np.ndarray, rv_series: np.ndarray):
        """Fit EWT decomposition and component models."""
        # Detect boundaries on returns
        self.boundaries = detect_boundaries_returns(returns, self.n_components)
        
        # Decompose returns
        components = ewt_decompose_returns(returns, self.boundaries, self.gamma)
        
        # For each component, fit appropriate model based on frequency
        self.component_models = []
        for i, comp in enumerate(components):
            # Compute component's realized variance
            comp_rv = comp**2
            
            # Choose model based on component characteristics
            comp_std = np.std(comp)
            comp_acf = np.corrcoef(comp[:-1], comp[1:])[0, 1] if len(comp) > 1 else 0
            
            if comp_std < 1e-6:  # Near-constant component
                model = {'type': 'constant', 'value': np.mean(comp_rv)}
            elif np.abs(comp_acf) > 0.1 and len(comp) > 10:  # Persistent component
                # AR(1) on log-variance for stable forecasts
                log_rv = np.log(np.maximum(comp_rv, 1e-8))
                try:
                    ar_model = ARIMA(log_rv, order=(1, 0, 0)).fit()
                    model = {'type': 'ar_log', 'model': ar_model}
                except:
                    model = {'type': 'persistence', 'value': comp_rv[-1]}
            else:  # High-frequency/noisy component
                model = {'type': 'garch', 'returns': comp}
            
            self.component_models.append(model)
        
        log.info(f"Fitted {len(self.component_models)} component models")
        return self
    
    def forecast_variance(self, recent_returns: np.ndarray) -> float:
        """Forecast variance by summing component variances."""
        if self.boundaries is None:
            raise ValueError("Model not fitted")
        
        # Decompose recent returns
        components = ewt_decompose_returns(recent_returns, self.boundaries, self.gamma)
        
        # Ensure we have enough components
        n_components = min(len(components), len(self.component_models))
        
        total_variance = 0.0
        
        for i in range(n_components):
            comp = components[i]
            model = self.component_models[i]
            
            if model['type'] == 'constant':
                var_forecast = model['value']
            elif model['type'] == 'ar_log':
                # Forecast log variance
                comp_rv = comp**2
                log_rv = np.log(np.maximum(comp_rv, 1e-8))
                log_forecast = model['model'].forecast(steps=1)[0]
                var_forecast = np.exp(log_forecast)
            elif model['type'] == 'persistence':
                var_forecast = model['value']
            elif model['type'] == 'garch':
                # Use recent component returns for GARCH forecast
                var_forecast = garch_forecast_robust(comp)
            else:
                # Fallback: historical variance
                var_forecast = np.var(comp[-22:]) if len(comp) >= 22 else np.var(comp)
            
            total_variance += max(var_forecast, 0.0)
        
        return max(total_variance, 1e-8)

# ======================================================
# Model Stacking
# ======================================================
class VolatilityModelStacker:
    """
    Linear stacking of multiple volatility models.
    Learns optimal weights on validation set.
    """
    
    def __init__(self, model_names: List[str]):
        self.model_names = model_names
        self.weights = None
        self.intercept = 0.0
        
    def fit(self, predictions: np.ndarray, targets: np.ndarray):
        """Learn optimal weights via constrained regression."""
        n_models = predictions.shape[1]
        
        # Ensure predictions are positive
        predictions = np.maximum(predictions, 1e-8)
        targets = np.maximum(targets, 1e-8)
        
        # Log transform for homoscedasticity
        log_pred = np.log(predictions)
        log_target = np.log(targets)
        
        # Constrained regression: weights sum to 1, non-negative
        def objective(w):
            w = w[:-1]  # Last element is intercept
            pred = log_pred @ w + w[-1]
            return np.mean((pred - log_target)**2)
        
        # Initial guess (equal weights)
        w0 = np.ones(n_models + 1) / (n_models + 1)
        
        # Constraints: weights non-negative, sum <= 1
        cons = ({'type': 'ineq', 'fun': lambda w: w[:-1]},  # Non-negative
                {'type': 'ineq', 'fun': lambda w: 1 - np.sum(w[:-1])})  # Sum <= 1
        
        result = minimize(objective, w0, constraints=cons, 
                         method='SLSQP', options={'maxiter': 100})
        
        if result.success:
            self.weights = result.x[:-1]
            self.intercept = result.x[-1]
            log.info(f"Stacking weights: {self.weights}, intercept: {self.intercept:.4f}")
        else:
            # Fallback to equal weights
            self.weights = np.ones(n_models) / n_models
            self.intercept = 0.0
            log.warning("Stacking optimization failed, using equal weights")
        
        return self
    
    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """Combine predictions using learned weights."""
        if self.weights is None:
            raise ValueError("Stacker not fitted")
        
        predictions = np.maximum(predictions, 1e-8)
        log_pred = np.log(predictions)
        
        # Weighted combination in log space
        log_combined = log_pred @ self.weights + self.intercept
        combined = np.exp(log_combined)
        
        return np.maximum(combined, 1e-8)

# ======================================================
# Main Forecasting Pipeline
# ======================================================
def rolling_oos_forecast(returns: np.ndarray, train_size: int = 1000, 
                        test_size: int = 500, window: int = 500) -> Dict:
    """
    Rolling out-of-sample forecasting pipeline.
    """
    # Precompute realized volatility
    rv_daily = realized_volatility(returns, window=1)
    
    # Train-test split
    train_returns = returns[:train_size]
    train_rv = rv_daily[:train_size]
    
    test_returns = returns[train_size:train_size + test_size]
    test_rv = rv_daily[train_size:train_size + test_size]
    
    log.info(f"Train: {len(train_returns)}, Test: {len(test_returns)}")
    
    # Initialize EWT model on training data
    ewt_model = EWTVolatilityForecaster(n_components=4, gamma=0.1)
    ewt_model.fit(train_returns, train_rv)
    
    # Rolling forecasts
    true_vol = []
    forecasts = {
        'persistence': [],
        'har_rv': [],
        'garch': [],
        'ewt': [],
        'stacked': []
    }
    
    # Validation set for stacking weights (last part of training window)
    val_predictions = []
    val_targets = []
    
    for i in range(window, len(test_returns)):
        # Get rolling window data
        start_idx = train_size + i - window
        ret_window = returns[start_idx:train_size + i]
        rv_window = rv_daily[start_idx:train_size + i]
        
        # True value (next period volatility)
        true_val = test_rv[i] if i < len(test_rv) else rv_daily[train_size + i]
        true_vol.append(true_val)
        
        # Individual model forecasts
        f_persistence = persistence_forecast(rv_window)
        f_har = har_rv_forecast(rv_window)
        f_garch = np.sqrt(garch_forecast_robust(ret_window))
        f_ewt = np.sqrt(ewt_model.forecast_variance(ret_window))
        
        forecasts['persistence'].append(f_persistence)
        forecasts['har_rv'].append(f_har)
        forecasts['garch'].append(f_garch)
        forecasts['ewt'].append(f_ewt)
        
        # Collect validation data for stacking (first half of test)
        if i < window + 100 and i >= window:
            val_predictions.append([f_persistence, f_har, f_garch, f_ewt])
            val_targets.append(true_val)
        
        # Progress logging
        if i % 100 == 0:
            log.info(f"Processed {i - window}/{len(test_returns) - window} test points")
    
    # Convert to arrays
    true_vol = np.array(true_vol)
    for key in forecasts:
        forecasts[key] = np.array(forecasts[key])
    
    # Stack models using validation set
    if len(val_predictions) > 20:
        val_pred_array = np.array(val_predictions)
        val_target_array = np.array(val_targets)
        
        stacker = VolatilityModelStacker(['persistence', 'har_rv', 'garch', 'ewt'])
        stacker.fit(val_pred_array, val_target_array)
        
        # Apply stacking to all forecasts
        all_preds = np.column_stack([forecasts['persistence'], 
                                    forecasts['har_rv'], 
                                    forecasts['garch'], 
                                    forecasts['ewt']])
        forecasts['stacked'] = stacker.predict(all_preds)
    else:
        # Simple average if not enough validation data
        forecasts['stacked'] = np.mean([forecasts['persistence'], 
                                       forecasts['har_rv'], 
                                       forecasts['garch'], 
                                       forecasts['ewt']], axis=0)
    
    return true_vol, forecasts

# ======================================================
# Evaluation and Metrics
# ======================================================
def evaluate_forecasts(true_vol: np.ndarray, forecasts: Dict) -> pd.DataFrame:
    """Calculate evaluation metrics."""
    results = []
    
    for model_name, pred in forecasts.items():
        if len(pred) != len(true_vol):
            log.warning(f"Mismatch: {model_name} has {len(pred)} predictions, expected {len(true_vol)}")
            continue
        
        # Calculate metrics
        mse = mean_squared_error(true_vol, pred)
        mae = mean_absolute_error(true_vol, pred)
        
        # Directional accuracy
        true_chg = true_vol[1:] - true_vol[:-1]
        pred_chg = pred[1:] - pred[:-1]
        if len(true_chg) > 0:
            direction_acc = np.mean((true_chg * pred_chg) > 0)
        else:
            direction_acc = np.nan
        
        # Relative MSE vs persistence
        mse_persistence = mean_squared_error(true_vol, forecasts['persistence'])
        rel_mse = mse / mse_persistence if mse_persistence > 0 else np.nan
        
        results.append({
            'Model': model_name,
            'MSE': mse,
            'MAE': mae,
            'Direction_Accuracy': direction_acc,
            'Rel_MSE_vs_Persistence': rel_mse,
            'Win_vs_Persistence': rel_mse < 1.0 if not np.isnan(rel_mse) else False
        })
    
    return pd.DataFrame(results)

def diebold_mariano_test(true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> Tuple[float, float]:
    """
    Simple Diebold-Mariano test for equal predictive accuracy.
    Returns test statistic and p-value (two-sided).
    """
    # Loss differential
    loss1 = (true - pred1)**2
    loss2 = (true - pred2)**2
    d = loss1 - loss2
    
    # Test statistic
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1) / len(d)
    
    if d_var <= 0:
        return 0.0, 1.0
    
    t_stat = d_mean / np.sqrt(d_var)
    
    # Approximate p-value using normal distribution
    from scipy import stats
    p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))
    
    return t_stat, p_value

# ======================================================
# Main Execution
# ======================================================
if __name__ == "__main__":
    # Load and prepare data
    try:
        # For demonstration, create synthetic data if file not found
        log.info("Generating synthetic volatility data for demonstration...")
        np.random.seed(42)
        n_points = 2000
        
        # Create realistic volatility process with clustering
        true_vol = np.zeros(n_points)
        true_vol[0] = 0.01
        for t in range(1, n_points):
            true_vol[t] = (0.05 + 0.85 * true_vol[t-1] + 
                          0.1 * np.random.randn()**2)
        
        # Generate returns with stochastic volatility
        returns = true_vol * np.random.randn(n_points)
        
        # Add some jumps
        jump_times = np.random.choice(n_points, size=20, replace=False)
        returns[jump_times] += 0.05 * np.sign(np.random.randn(20))
        
    except Exception as e:
        log.error(f"Data loading failed: {e}")
        raise
    
    # Run forecasting pipeline
    log.info("Starting rolling out-of-sample forecast...")
    true_vol, forecasts = rolling_oos_forecast(
        returns, 
        train_size=1000,
        test_size=500,
        window=400
    )
    
    # Evaluate results
    results_df = evaluate_forecasts(true_vol, forecasts)
    log.info("\n" + "="*60)
    log.info("FORECASTING RESULTS (Volatility Space)")
    log.info("="*60)
    print(results_df.to_string(float_format=lambda x: f"{x:.6f}"))
    
    # Diebold-Mariano tests vs persistence
    log.info("\n" + "="*60)
    log.info("DIEBOLD-MARIANO TESTS vs PERSISTENCE")
    log.info("="*60)
    
    persistence_pred = forecasts['persistence']
    for model_name, pred in forecasts.items():
        if model_name != 'persistence' and len(pred) == len(persistence_pred):
            t_stat, p_val = diebold_mariano_test(true_vol, pred, persistence_pred)
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            log.info(f"{model_name:12s}: t={t_stat:7.3f}, p={p_val:6.4f} {sig}")
    
    # Identify best model
    best_idx = results_df['MSE'].idxmin()
    best_model = results_df.loc[best_idx, 'Model']
    best_mse = results_df.loc[best_idx, 'MSE']
    
    log.info("\n" + "="*60)
    log.info(f"BEST MODEL: {best_model} (MSE: {best_mse:.6f})")
    
    # Check if we beat persistence
    persistence_mse = results_df[results_df['Model'] == 'persistence']['MSE'].values[0]
    if best_mse < persistence_mse:
        improvement = 100 * (persistence_mse - best_mse) / persistence_mse
        log.info(f"SUCCESS: Beat persistence by {improvement:.2f}% in MSE")
    else:
        log.warning("FAILED: Did not beat persistence benchmark")
    
    log.info("="*60)