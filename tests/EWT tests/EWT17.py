import numpy as np
import pandas as pd
import logging
import warnings
import traceback
from typing import Tuple, Dict, List, Optional

from numpy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks
from scipy.optimize import minimize
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.linear_model import OLS
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ======================================================
# Logging
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger()

# ======================================================
# Smooth Transition Function
# ======================================================
def smooth_beta(x: np.ndarray) -> np.ndarray:
    """Smooth transition function for EWT boundaries."""
    x = np.clip(x, 0.0, 1.0)
    return x**4 * (35 - 84*x + 70*x**2 - 20*x**3)

# ======================================================
# EWT Boundary Detection (TRAIN ONLY)
# ======================================================
def detect_boundaries(signal: np.ndarray, max_modes: int = 5) -> np.ndarray:
    """
    Detect boundaries for EWT decomposition.
    Applied to returns, not realized variance.
    """
    N = len(signal)
    if N < 32:
        raise ValueError(f"Signal too short for boundary detection: {N} < 32")
    
    # Remove mean for better spectral analysis
    signal = signal - np.mean(signal)
    
    spectrum = np.abs(fft(signal))[:N // 2]
    spectrum[0] = 0.0  # remove DC
    
    # Find peaks with appropriate distance
    min_distance = max(1, (N // 2) // (max_modes * 4))
    peaks, properties = find_peaks(
        spectrum, 
        distance=min_distance,
        prominence=np.percentile(spectrum, 75)
    )
    
    if len(peaks) < 2:
        # Fallback to uniform frequency bands
        log.warning("Not enough spectral peaks, using uniform boundaries")
        return np.linspace(0.1, 0.4, max_modes-1)
    
    # Sort peaks by prominence (if available) or magnitude
    if 'prominences' in properties:
        prominences = properties['prominences']
        sorted_idx = np.argsort(prominences)[::-1][:min(max_modes, len(peaks))]
    else:
        sorted_idx = np.argsort(spectrum[peaks])[::-1][:min(max_modes, len(peaks))]
    
    selected_peaks = peaks[np.sort(sorted_idx)]
    
    # Convert to normalized frequencies
    freqs = selected_peaks / float(N)
    
    # Create boundaries as midpoints between peaks
    boundaries = 0.5 * (freqs[:-1] + freqs[1:])
    
    # Ensure boundaries are within (0, 0.5)
    boundaries = np.clip(boundaries, 0.05, 0.45)
    
    log.info(f"Detected {len(boundaries)+1} frequency bands")
    return boundaries

# ======================================================
# EWT Decomposition
# ======================================================
def ewt_decompose(signal: np.ndarray, boundaries: np.ndarray, 
                  gamma: float = 0.1) -> List[np.ndarray]:
    """
    Empirical Wavelet Transform decomposition.
    Returns orthogonal components of the input signal.
    """
    N = len(signal)
    signal_mean = np.mean(signal)
    signal_centered = signal - signal_mean
    
    # Frequency axis
    freqs = np.abs(fftfreq(N))
    fft_sig = fft(signal_centered)
    
    # Sort and validate boundaries
    b = np.sort(np.asarray(boundaries))
    b = np.clip(b, 1e-12, 0.5 - 1e-12)
    
    filters = []
    
    # Low-pass filter (first band)
    w1 = b[0]
    phi = np.zeros(N, dtype=complex)
    low_mask = freqs <= (1 - gamma) * w1
    trans_mask = ((1 - gamma) * w1 < freqs) & (freqs <= (1 + gamma) * w1)
    phi[low_mask] = 1.0
    if np.any(trans_mask):
        phi[trans_mask] = np.cos(
            (np.pi / 2) * smooth_beta(
                (freqs[trans_mask] - (1 - gamma) * w1) / (2 * gamma * w1)
            )
        )
    filters.append(phi)
    
    # Band-pass filters
    for i in range(len(b) - 1):
        w0, w1 = b[i], b[i + 1]
        psi = np.zeros(N, dtype=complex)
        
        # Central band
        band_mask = ((1 + gamma) * w0 <= freqs) & (freqs <= (1 - gamma) * w1)
        psi[band_mask] = 1.0
        
        # Upper transition
        up_mask = ((1 - gamma) * w1 < freqs) & (freqs <= (1 + gamma) * w1)
        if np.any(up_mask):
            psi[up_mask] = np.cos(
                (np.pi / 2) * smooth_beta(
                    (freqs[up_mask] - (1 - gamma) * w1) / (2 * gamma * w1)
                )
            )
        
        # Lower transition
        down_mask = ((1 - gamma) * w0 <= freqs) & (freqs <= (1 + gamma) * w0)
        if np.any(down_mask):
            psi[down_mask] = np.sin(
                (np.pi / 2) * smooth_beta(
                    (freqs[down_mask] - (1 - gamma) * w0) / (2 * gamma * w0)
                )
            )
        
        filters.append(psi)
    
    # High-pass filter (last band)
    w_last = b[-1]
    psi_last = np.zeros(N, dtype=complex)
    high_mask = freqs >= (1 + gamma) * w_last
    trans_last_mask = ((1 - gamma) * w_last <= freqs) & (freqs <= (1 + gamma) * w_last)
    psi_last[high_mask] = 1.0
    if np.any(trans_last_mask):
        psi_last[trans_last_mask] = np.sin(
            (np.pi / 2) * smooth_beta(
                (freqs[trans_last_mask] - (1 - gamma) * w_last) / (2 * gamma * w_last)
            )
        )
    filters.append(psi_last)
    
    # Reconstruct components
    components = []
    for f in filters:
        comp = np.real(ifft(fft_sig * f)) + signal_mean / len(filters)
        components.append(comp)
    
    return components

# ======================================================
# Realized Variance Calculation
# ======================================================
def realized_variance(returns: np.ndarray, window: int = 1) -> np.ndarray:
    """
    Calculate realized variance.
    For daily RV, use window=1 (squared returns).
    For better estimation, can use intraday data but here we use daily.
    """
    rv = returns ** 2
    if window > 1:
        # Rolling sum of squared returns
        rv = pd.Series(rv).rolling(window, min_periods=1).sum().values
    return rv

def calculate_har_features(rv: np.ndarray) -> pd.DataFrame:
    """
    Calculate features for HAR-RV model:
    - Daily RV (RV_d)
    - Weekly RV (average of last 5 days, RV_w)
    - Monthly RV (average of last 22 days, RV_m)
    """
    rv_series = pd.Series(rv)
    
    features = pd.DataFrame({
        'RV_d': rv_series,
        'RV_w': rv_series.rolling(5, min_periods=1).mean(),
        'RV_m': rv_series.rolling(22, min_periods=1).mean()
    })
    
    # Shift features for forecasting (using only past information)
    features = features.shift(1)
    
    return features

# ======================================================
# Benchmark Models
# ======================================================
class VolatilityForecasters:
    """Collection of volatility forecasting models with robust error handling."""
    
    @staticmethod
    def persistence_forecast(rv: np.ndarray) -> float:
        """Simple persistence: RV_t+1 = RV_t"""
        try:
            last_rv = float(rv[-1])
            return max(last_rv, 1e-12)
        except:
            return 1e-12
    
    @staticmethod
    def har_rv_forecast(rv: np.ndarray, har_model: Optional[OLS] = None) -> Tuple[float, Optional[OLS]]:
        """
        HAR-RV model: log(RV_t+1) = β0 + β1 log(RV_d) + β2 log(RV_w) + β3 log(RV_m)
        Returns forecast and fitted model.
        """
        try:
            # Calculate features
            features = calculate_har_features(rv)
            
            # Remove rows with NaN (beginning of series)
            valid_idx = features.dropna().index
            if len(valid_idx) < 23:  # Need at least 22 for monthly + 1 for regression
                return VolatilityForecasters.persistence_forecast(rv), None
            
            X = features.iloc[valid_idx]
            y = rv[valid_idx]
            
            # Transform to log space for stability
            log_y = np.log(y + 1e-12)
            log_X = np.log(X + 1e-12)
            
            # Fit HAR model if not provided
            if har_model is None:
                har_model = OLS(log_y, log_X).fit()
            
            # Prepare features for forecast
            last_features = pd.DataFrame({
                'RV_d': [rv[-1]],
                'RV_w': [np.mean(rv[-5:])],
                'RV_m': [np.mean(rv[-22:])]
            })
            last_log_features = np.log(last_features + 1e-12)
            
            # Forecast
            log_forecast = har_model.predict(last_log_features)[0]
            forecast = np.exp(log_forecast)
            
            return max(float(forecast), 1e-12), har_model
            
        except Exception as e:
            log.warning(f"HAR-RV forecast failed: {e}")
            return VolatilityForecasters.persistence_forecast(rv), None
    
    @staticmethod
    def garch_forecast(returns: np.ndarray) -> float:
        """GARCH(1,1) forecast on returns."""
        try:
            if len(returns) < 50:
                # Need sufficient data for GARCH
                sample_var = np.nanvar(returns)
                return max(float(sample_var), 1e-12)
            
            # Fit GARCH(1,1) with Student's t distribution
            model = arch_model(
                returns, 
                mean='Zero', 
                vol='GARCH', 
                p=1, 
                q=1, 
                dist='t',
                rescale=False
            )
            
            # Suppress convergence warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = model.fit(disp='off', show_warning=False)
            
            # 1-step ahead forecast
            forecast = res.forecast(horizon=1)
            variance = forecast.variance.values[-1, 0]
            
            return max(float(variance), 1e-12)
            
        except Exception as e:
            log.warning(f"GARCH forecast failed: {e}")
            sample_var = np.nanvar(returns[-50:]) if len(returns) >= 50 else np.nanvar(returns)
            return max(float(sample_var), 1e-12)
    
    @staticmethod
    def arima_rv_forecast(rv: np.ndarray) -> float:
        """ARIMA(1,0,1) on log(RV) for stability."""
        try:
            if len(rv) < 20:
                return VolatilityForecasters.persistence_forecast(rv)
            
            # Work in log space for stability
            log_rv = np.log(rv + 1e-12)
            
            model = ARIMA(log_rv, order=(1, 0, 1))
            res = model.fit()
            
            log_forecast = res.forecast(steps=1)[0]
            forecast = np.exp(log_forecast)
            
            return max(float(forecast), 1e-12)
            
        except Exception as e:
            log.warning(f"ARIMA forecast failed: {e}")
            return VolatilityForecasters.persistence_forecast(rv)

# ======================================================
# EWT-Based Volatility Model
# ======================================================
class EWTVolatilityModel:
    """
    EWT-based volatility forecasting model.
    Decomposes returns into frequency components, forecasts each component's
    variance, and recombines.
    """
    
    def __init__(self, max_modes: int = 5, gamma: float = 0.1):
        self.max_modes = max_modes
        self.gamma = gamma
        self.boundaries = None
        self.ewt_components = None
        
    def fit(self, returns: np.ndarray) -> None:
        """Detect EWT boundaries on training returns."""
        try:
            self.boundaries = detect_boundaries(returns, self.max_modes)
            log.info(f"Fitted EWT with {len(self.boundaries)+1} frequency bands")
        except Exception as e:
            log.error(f"EWT boundary detection failed: {e}")
            # Fallback to uniform boundaries
            self.boundaries = np.linspace(0.1, 0.4, self.max_modes-1)
    
    def forecast(self, returns: np.ndarray, rv: np.ndarray) -> float:
        """
        Forecast volatility using EWT decomposition.
        
        Steps:
        1. Decompose recent returns into frequency components
        2. For each component, forecast its variance using AR(1)
        3. Sum component variances (orthogonality ensures additivity)
        """
        if self.boundaries is None or len(returns) < 100:
            # Fallback to HAR if insufficient data
            forecast, _ = VolatilityForecasters.har_rv_forecast(rv)
            return forecast
        
        try:
            # Decompose the most recent returns (use last 252 days for stability)
            lookback = min(252, len(returns))
            recent_returns = returns[-lookback:]
            
            # Decompose returns (not RV!)
            components = ewt_decompose(recent_returns, self.boundaries, self.gamma)
            
            # Forecast variance for each component
            component_variances = []
            for comp in components:
                if np.std(comp) < 1e-8:
                    # Near-constant component
                    comp_var = np.var(comp)
                    component_variances.append(comp_var)
                    continue
                
                # Use AR(1) on squared component (proxy for component variance)
                comp_squared = comp ** 2
                
                try:
                    # Simple exponential smoothing as fallback
                    alpha = 0.94  # RiskMetrics decay factor
                    weights = np.array([(1-alpha) * (alpha**i) 
                                       for i in range(len(comp_squared)-1, -1, -1)])
                    weights = weights / weights.sum()
                    
                    comp_var_forecast = np.sum(weights * comp_squared)
                    component_variances.append(comp_var_forecast)
                except:
                    # Fallback to recent variance
                    comp_var_forecast = np.var(comp[-20:])
                    component_variances.append(comp_var_forecast)
            
            # Total variance forecast is sum of component variances
            total_variance = np.sum(component_variances)
            
            return max(float(total_variance), 1e-12)
            
        except Exception as e:
            log.warning(f"EWT forecast failed: {e}")
            # Fallback to persistence
            return VolatilityForecasters.persistence_forecast(rv)

# ======================================================
# Hybrid Model with Stacking
# ======================================================
class StackedVolatilityModel:
    """
    Stacked ensemble model that combines multiple forecasts using OLS.
    Weights are learned on a rolling training window.
    """
    
    def __init__(self, model_names: List[str] = None):
        self.model_names = model_names or ['HAR', 'GARCH', 'EWT', 'Persistence']
        self.weights = None
        self.intercept = 0.0
    
    def fit_weights(self, forecasts: np.ndarray, true_rv: np.ndarray) -> None:
        """
        Learn optimal weights for combining forecasts using OLS.
        forecasts shape: (n_periods, n_models)
        true_rv shape: (n_periods,)
        """
        try:
            # Add intercept
            X = np.column_stack([np.ones(len(forecasts)), forecasts])
            
            # Fit OLS
            coeffs = np.linalg.lstsq(X, true_rv, rcond=None)[0]
            
            self.intercept = coeffs[0]
            self.weights = coeffs[1:]
            
            # Ensure weights sum to 1 (convex combination)
            if np.sum(self.weights) > 0:
                self.weights = self.weights / np.sum(self.weights)
            
            log.info(f"Stacked model weights: {dict(zip(self.model_names, self.weights))}")
            
        except Exception as e:
            log.warning(f"Weight learning failed: {e}, using equal weights")
            self.weights = np.ones(len(self.model_names)) / len(self.model_names)
            self.intercept = 0.0
    
    def predict(self, forecasts: np.ndarray) -> float:
        """Combine forecasts using learned weights."""
        if self.weights is None:
            # Use equal weights if not trained
            combined = np.mean(forecasts)
        else:
            combined = self.intercept + np.dot(forecasts, self.weights)
        
        return max(float(combined), 1e-12)

# ======================================================
# Main Forecasting Pipeline
# ======================================================
def run_volatility_forecast_pipeline(filepath: str, window_size: int = 1000) -> Dict:
    """
    Main pipeline for volatility forecasting.
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file with 'close' prices
    window_size : int
        Rolling window size for model estimation
    
    Returns:
    --------
    Dictionary with forecasts and evaluation metrics
    """
    # Load and prepare data
    log.info("Loading data...")
    df = pd.read_csv(filepath)
    
    if 'close' not in df.columns:
        raise ValueError("Data must contain 'close' column")
    
    # Calculate returns
    df['returns'] = np.log(df['close']).diff()
    df.dropna(inplace=True)
    
    # Calculate realized variance (daily squared returns)
    df['rv'] = realized_variance(df['returns'].values, window=1)
    
    returns = df['returns'].values
    rv = df['rv'].values
    
    # Train/test split (70/30)
    split_idx = int(len(rv) * 0.7)
    
    train_returns = returns[:split_idx]
    train_rv = rv[:split_idx]
    test_returns = returns[split_idx:]
    test_rv = rv[split_idx:]
    
    log.info(f"Train size: {len(train_rv)}, Test size: {len(test_rv)}")
    
    # Initialize models
    ewt_model = EWTVolatilityModel(max_modes=5, gamma=0.1)
    
    # Fit EWT boundaries on training returns
    ewt_model.fit(train_returns)
    
    # Initialize stacked model
    stacked_model = StackedVolatilityModel(
        model_names=['Persistence', 'HAR', 'GARCH', 'EWT', 'ARIMA']
    )
    
    # Rolling out-of-sample forecasts
    forecasts = {
        'Persistence': [],
        'HAR': [],
        'GARCH': [],
        'EWT': [],
        'ARIMA': [],
        'Stacked': []
    }
    true_values = []
    
    # Buffer for training the stacked model weights
    weight_training_buffer = []
    
    log.info("Starting out-of-sample forecasting...")
    
    for i in range(window_size, len(test_rv)):
        try:
            # Define estimation window (fixed length)
            start_idx = i - window_size
            end_idx = i
            
            window_returns = test_returns[start_idx:end_idx]
            window_rv = test_rv[start_idx:end_idx]
            
            # True value to forecast
            true_rv = test_rv[i]
            true_values.append(true_rv)
            
            # Generate individual forecasts
            persistence_fc = VolatilityForecasters.persistence_forecast(window_rv)
            har_fc, _ = VolatilityForecasters.har_rv_forecast(window_rv)
            garch_fc = VolatilityForecasters.garch_forecast(window_returns)
            ewt_fc = ewt_model.forecast(window_returns, window_rv)
            arima_fc = VolatilityForecasters.arima_rv_forecast(window_rv)
            
            # Store individual forecasts
            forecasts['Persistence'].append(persistence_fc)
            forecasts['HAR'].append(har_fc)
            forecasts['GARCH'].append(garch_fc)
            forecasts['EWT'].append(ewt_fc)
            forecasts['ARIMA'].append(arima_fc)
            
            # Prepare forecasts for stacking
            current_forecasts = np.array([
                persistence_fc, har_fc, garch_fc, ewt_fc, arima_fc
            ])
            
            # Update weight training buffer
            if len(weight_training_buffer) < 100:
                # Collect data for weight training
                weight_training_buffer.append({
                    'forecasts': current_forecasts,
                    'true': true_rv
                })
            else:
                # Train weights every 100 periods
                if len(weight_training_buffer) == 100:
                    train_fc = np.array([x['forecasts'] for x in weight_training_buffer])
                    train_true = np.array([x['true'] for x in weight_training_buffer])
                    stacked_model.fit_weights(train_fc, train_true)
            
            # Generate stacked forecast
            if stacked_model.weights is not None:
                stacked_fc = stacked_model.predict(current_forecasts)
            else:
                # Use simple average before weights are learned
                stacked_fc = np.mean(current_forecasts)
            
            forecasts['Stacked'].append(stacked_fc)
            
            if (i - window_size) % 100 == 0:
                log.info(f"Progress: {i-window_size+1}/{len(test_rv)-window_size}")
                
        except Exception as e:
            log.error(f"Error at iteration {i}: {e}")
            traceback.print_exc()
            # Append NaN for failed forecasts
            for key in forecasts:
                forecasts[key].append(np.nan)
            true_values.append(np.nan)
    
    # Clean up results (remove NaN values)
    valid_idx = ~np.isnan(true_values)
    true_values_clean = np.array(true_values)[valid_idx]
    
    for key in forecasts:
        forecasts[key] = np.array(forecasts[key])[valid_idx]
    
    # Convert variance forecasts to volatility
    true_vol = np.sqrt(np.clip(true_values_clean, 0, None))
    
    forecast_vol = {}
    for key in forecasts:
        forecast_vol[key] = np.sqrt(np.clip(forecasts[key], 0, None))
    
    # Calculate evaluation metrics
    results = {}
    metrics = {}
    
    for model_name, pred_vol in forecast_vol.items():
        if len(pred_vol) == 0:
            metrics[model_name] = {'MSE': np.nan, 'MAE': np.nan}
            continue
        
        mse = mean_squared_error(true_vol, pred_vol)
        mae = mean_absolute_error(true_vol, pred_vol)
        
        metrics[model_name] = {'MSE': mse, 'MAE': mae}
        
        log.info(f"{model_name:12} | MSE: {mse:.6f} | MAE: {mae:.6f}")
    
    # Diebold-Mariano test (simple version)
    log.info("\n" + "="*50)
    log.info("Relative Performance (vs Persistence):")
    
    baseline_mse = metrics['Persistence']['MSE']
    for model_name, model_metrics in metrics.items():
        if model_name != 'Persistence' and not np.isnan(model_metrics['MSE']):
            improvement = (baseline_mse - model_metrics['MSE']) / baseline_mse * 100
            log.info(f"{model_name:12} | MSE Improvement: {improvement:.2f}%")
    
    # Prepare final results
    results['metrics'] = metrics
    results['true_vol'] = true_vol
    results['forecast_vol'] = forecast_vol
    results['forecasts'] = forecasts
    
    return results

# ======================================================
# Main Execution
# ======================================================
if __name__ == "__main__":
    # Example usage
    try:
        # Update this path to your data
        data_path = r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv"
        
        results = run_volatility_forecast_pipeline(
            filepath=data_path,
            window_size=1000
        )
        
        log.info("\n" + "="*50)
        log.info("Pipeline completed successfully")
        log.info("Best model by MSE: " + 
                min(results['metrics'].items(), 
                    key=lambda x: x[1]['MSE'] if not np.isnan(x[1]['MSE']) else np.inf)[0])
        
    except Exception as e:
        log.error(f"Pipeline failed: {e}")
        traceback.print_exc()