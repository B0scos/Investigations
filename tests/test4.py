import numpy as np
import pandas as pd
import scipy.fft as fft
import warnings
from scipy.linalg import pinv

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ======================================================
# CONFIG
# ======================================================

CSV_PATH = r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv"
PRICE_COL = "close"
DATE_COL = "timestamp"

VOL_WINDOW = 30
TRAIN_RATIO = 0.90
EWT_ALPHA = 0.25

ARIMA_ORDER = (1, 1, 1)
GARCH_P, GARCH_Q = 1, 1

# ELM parameters
ELM_HIDDEN_SIZE = 20
LOOKBACK = 10  # How many past values to use as input

import numpy as np
import pandas as pd
import scipy.fft as fft
import warnings

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ======================================================
# CONFIG
# ======================================================

CSV_PATH = r"C:\Users\ferre\OneDrive\Área de Trabalho\quant\data\BTC_data.csv"
PRICE_COL = "close"
DATE_COL = "timestamp"

VOL_WINDOW = 30
TRAIN_RATIO = 0.90
EWT_ALPHA = 0.25

ARIMA_ORDER = (1, 1, 1)
GARCH_P, GARCH_Q = 1, 1

# ======================================================
# Metrics
# ======================================================

def rmse(y, yhat):
    return np.sqrt(mean_squared_error(y, yhat))

# ======================================================
# Data prep
# ======================================================

def load_volatility():
    df = pd.read_csv(CSV_PATH, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL)

    r = np.log(df[PRICE_COL] / df[PRICE_COL].shift(1)) 
    vol = r.rolling(VOL_WINDOW).std()

    return vol.dropna().values

# ======================================================
# EWT decomposition
# ======================================================

from scipy.signal import find_peaks

def ewt_decompose(signal, alpha=0.25, min_peaks=3):
    N = len(signal)

    # FFT
    X = np.abs(fft.fft(signal))
    freqs = fft.fftfreq(N)

    mask = freqs > 0
    X, freqs = X[mask], freqs[mask]

    # Normalize
    Xn = (X - X.min()) / (X.max() - X.min() + 1e-12)

    # --------------------------------------------------
    # 1. Peak detection (robust)
    # --------------------------------------------------
    peaks, props = find_peaks(Xn, height=alpha)

    # If too few peaks → relax threshold
    if len(peaks) < min_peaks:
        peaks, props = find_peaks(Xn, height=np.quantile(Xn, 0.75))

    # Still too few → fallback to strongest frequencies
    if len(peaks) < min_peaks:
        peaks = np.argsort(Xn)[-min_peaks:]

    peak_freqs = np.sort(freqs[peaks])

    # --------------------------------------------------
    # 2. Boundary construction
    # --------------------------------------------------
    bounds = 0.5 * (peak_freqs[:-1] + peak_freqs[1:])

    # Normalize to [0, π]
    bounds = np.pi * bounds / bounds.max()

    return build_ewt_bands(signal, bounds)


def build_ewt_bands(signal, bounds):
    N = len(signal)
    X = fft.fft(signal)
    freqs = np.linspace(0, np.pi, N // 2 + 1)

    bands = []

    def mirror(H):
        full = np.zeros(N)
        full[:len(H)] = H
        full[len(H):] = H[-2:0:-1]
        return full

    # Low-pass
    H = np.zeros_like(freqs)
    H[freqs <= bounds[0]] = 1
    bands.append(np.real(fft.ifft(X * mirror(H))))

    # Band-pass
    for i in range(len(bounds) - 1):
        H = np.zeros_like(freqs)
        H[(freqs >= bounds[i]) & (freqs <= bounds[i + 1])] = 1
        bands.append(np.real(fft.ifft(X * mirror(H))))

    # High-pass
    H = np.zeros_like(freqs)
    H[freqs >= bounds[-1]] = 1
    bands.append(np.real(fft.ifft(X * mirror(H))))

    return bands

# ======================================================
# Band diagnostics
# ======================================================

def band_is_heteroskedastic(x):
    try:
        pval = het_arch(x)[1]
        return pval < 0.05
    except:
        return False

# ======================================================
# Forecasting per band
# ======================================================

def forecast_band(train, h, use_garch):
    if use_garch:
        model = arch_model(train, mean="Zero", vol="GARCH",
                           p=GARCH_P, q=GARCH_Q)
        fit = model.fit(disp="off")
        f = fit.forecast(horizon=h).variance.values[-1]
        return np.sqrt(f)
    else:
        model = ARIMA(train, order=ARIMA_ORDER)
        fit = model.fit()
        return fit.forecast(h)

# ======================================================
# Main experiment
# ======================================================

def main():
    vol = load_volatility()
    n = len(vol)
    split = int(n * TRAIN_RATIO)

    train, test = vol[:split], vol[split:]
    H = len(test)

    # -------------------------
    # Baseline: GARCH
    # -------------------------
    base_model = arch_model(train, mean="Zero", vol="GARCH",
                            p=1, q=1)
    base_fit = base_model.fit(disp="off")
    base_fcst = np.sqrt(base_fit.forecast(horizon=H).variance.values[-1])

    # -------------------------
    # ABS-VM
    # -------------------------
    bands = ewt_decompose(train, alpha=EWT_ALPHA)

    band_forecasts = []

    for b in bands:
        use_garch = band_is_heteroskedastic(b)
        band_forecasts.append(forecast_band(b, H, use_garch))

    abs_vm_forecast = np.sum(band_forecasts, axis=0)

    # -------------------------
    # Results
    # -------------------------
    print("\n===== RESULTS =====")
    print("Baseline GARCH RMSE :", rmse(test, base_fcst))
    print("ABS-VM RMSE         :", rmse(test, abs_vm_forecast))

if __name__ == "__main__":
    main()


# ======================================================
# ELM Implementation
# ======================================================

class SimpleELM:
    """Basic Extreme Learning Machine"""
    def __init__(self, n_hidden=20):
        self.n_hidden = n_hidden
        self.W = None  # Input weights
        self.b = None  # Biases
        self.beta = None  # Output weights
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def fit(self, X, y):
        """Train ELM in one step"""
        n_samples, n_features = X.shape
        
        # 1. Randomly initialize input weights and biases
        self.W = np.random.randn(n_features, self.n_hidden)
        self.b = np.random.randn(self.n_hidden)
        
        # 2. Calculate hidden layer output
        H = self.sigmoid(X @ self.W + self.b)  # (n_samples, n_hidden)
        
        # 3. Calculate output weights analytically (Moore-Penrose pseudo-inverse)
        self.beta = pinv(H) @ y.reshape(-1, 1)
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        H = self.sigmoid(X @ self.W + self.b)
        return (H @ self.beta).flatten()
    
    def forecast(self, X_train, y_train, steps=1):
        """Forecast multiple steps ahead (iterative)"""
        predictions = []
        X_current = X_train[-1].reshape(1, -1)  # Last known input
        
        for _ in range(steps):
            pred = self.predict(X_current)[0]
            predictions.append(pred)
            
            # Update input window: shift and append prediction
            X_current = np.roll(X_current, -1)
            X_current[0, -1] = pred
        
        return np.array(predictions)


class PSOELM(SimpleELM):
    """ELM optimized with Particle Swarm Optimization (simpler than ABC)"""
    def __init__(self, n_hidden=20, n_particles=30, max_iter=50):
        super().__init__(n_hidden)
        self.n_particles = n_particles
        self.max_iter = max_iter
    
    def fit(self, X, y):
        """Optimize ELM parameters with PSO"""
        n_samples, n_features = X.shape
        
        # Initialize particles (each particle = [W, b])
        particles = []
        velocities = []
        personal_best = []
        personal_best_scores = []
        
        for _ in range(self.n_particles):
            W = np.random.randn(n_features, self.n_hidden) * 0.1
            b = np.random.randn(self.n_hidden) * 0.1
            particles.append([W, b])
            velocities.append([np.zeros_like(W), np.zeros_like(b)])
            personal_best.append([W.copy(), b.copy()])
            personal_best_scores.append(float('inf'))
        
        # Global best
        global_best = particles[0].copy()
        global_best_score = float('inf')
        
        # PSO main loop
        for iteration in range(self.max_iter):
            for i in range(self.n_particles):
                W, b = particles[i]
                
                # Build ELM with current particle
                self.W, self.b = W, b
                
                # Calculate hidden layer
                H = self.sigmoid(X @ W + b)
                beta = pinv(H) @ y.reshape(-1, 1)
                
                # Calculate error
                y_pred = (H @ beta).flatten()
                score = mean_squared_error(y, y_pred)
                
                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best[i] = [W.copy(), b.copy()]
                
                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best = [W.copy(), b.copy()]
            
            # Update velocities and positions
            w = 0.7  # Inertia weight
            c1, c2 = 1.5, 1.5  # Cognitive and social factors
            
            for i in range(self.n_particles):
                for j in range(2):  # For W and b
                    r1, r2 = np.random.rand(), np.random.rand()
                    
                    # Velocity update
                    velocities[i][j] = (w * velocities[i][j] + 
                                       c1 * r1 * (personal_best[i][j] - particles[i][j]) +
                                       c2 * r2 * (global_best[j] - particles[i][j]))
                    
                    # Position update
                    particles[i][j] += velocities[i][j]
        
        # Use global best parameters
        self.W, self.b = global_best
        H = self.sigmoid(X @ self.W + self.b)
        self.beta = pinv(H) @ y.reshape(-1, 1)
        
        return self


# ======================================================
# Data Preparation for ELM
# ======================================================

def prepare_elm_data(series, lookback=10):
    """Convert time series to supervised learning format"""
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i-lookback:i])
        y.append(series[i])
    return np.array(X), np.array(y)


# ======================================================
# Updated Forecasting Function
# ======================================================

def forecast_band(train, h, use_garch, use_elm=False, lookback=LOOKBACK):
    """Forecast a band with optional ELM model"""
    
    # Prepare data if using ELM
    if use_elm and len(train) > lookback + 10:
        try:
            # Prepare supervised data
            X, y = prepare_elm_data(train, lookback)
            
            # Split for training (last 'h' points for validation)
            X_train, y_train = X[:-h], y[:-h]
            
            # Create and train ELM (use PSOELM for better results)
            elm = PSOELM(n_hidden=ELM_HIDDEN_SIZE, n_particles=20, max_iter=30)
            elm.fit(X_train, y_train)
            
            # Forecast
            predictions = elm.forecast(X_train, y_train, steps=h)
            return predictions
            
        except Exception as e:
            print(f"ELM failed: {e}, falling back to default model")
            use_elm = False
    
    # Fall back to original models
    if use_garch:
        model = arch_model(train, mean="Zero", vol="GARCH",
                          p=GARCH_P, q=GARCH_Q)
        fit = model.fit(disp="off")
        f = fit.forecast(horizon=h).variance.values[-1]
        return np.sqrt(f)
    else:
        model = ARIMA(train, order=ARIMA_ORDER)
        fit = model.fit()
        return fit.forecast(h)


# ======================================================
# Band Type Detection
# ======================================================

def detect_band_type(band, lookback=LOOKBACK):
    """Determine which model to use for each band"""
    
    if len(band) < lookback + 20:
        return "arima"  # Too short for ELM
    
    # Check heteroskedasticity
    is_hetero = band_is_heteroskedastic(band)
    
    # Check linearity (simple test)
    from scipy import stats
    X = np.arange(len(band)).reshape(-1, 1)
    slope, _, r_value, _, _ = stats.linregress(X.flatten(), band)
    linearity = r_value ** 2  # R-squared
    
    # Decision logic
    if is_hetero:
        return "garch"
    elif linearity < 0.3:  # Low linearity → use ELM for nonlinear patterns
        return "elm"
    else:
        return "arima"


# ======================================================
# Updated Main Function
# ======================================================

def main():
    vol = load_volatility()
    n = len(vol)
    split = int(n * TRAIN_RATIO)

    train, test = vol[:split], vol[split:]
    H = len(test)

    # -------------------------
    # Baseline: GARCH
    # -------------------------
    print("Training baseline GARCH...")
    base_model = arch_model(train, mean="Zero", vol="GARCH",
                           p=1, q=1)
    base_fit = base_model.fit(disp="off")
    base_fcst = np.sqrt(base_fit.forecast(horizon=H).variance.values[-1])

    # -------------------------
    # Enhanced ABS-VM with ELM
    # -------------------------
    print("Decomposing with EWT...")
    bands = ewt_decompose(train, alpha=EWT_ALPHA)
    
    print(f"Decomposed into {len(bands)} bands")
    band_forecasts = []
    band_types = []

    for i, b in enumerate(bands):
        # Detect band type
        b_type = detect_band_type(b)
        band_types.append(b_type)
        
        print(f"Band {i}: {b_type.upper()}")
        
        # Forecast based on band type
        if b_type == "elm":
            fcst = forecast_band(b, H, use_garch=False, use_elm=True)
        elif b_type == "garch":
            fcst = forecast_band(b, H, use_garch=True, use_elm=False)
        else:  # arima
            fcst = forecast_band(b, H, use_garch=False, use_elm=False)
        
        band_forecasts.append(fcst)

    # Weighted combination (simple equal weights for now)
    weights = np.ones(len(bands)) / len(bands)
    enhanced_forecast = np.sum([w*f for w, f in zip(weights, band_forecasts)], axis=0)

    # -------------------------
    # Simple ELM baseline (for comparison)
    # -------------------------
    print("Training ELM baseline...")
    X_train, y_train = prepare_elm_data(train, LOOKBACK)
    elm_base = SimpleELM(n_hidden=ELM_HIDDEN_SIZE)
    elm_base.fit(X_train, y_train)
    
    # Prepare test data for ELM
    last_window = train[-LOOKBACK:]
    elm_fcst = elm_base.forecast(X_train, y_train, steps=H)

    # -------------------------
    # Results
    # -------------------------
    print("\n" + "="*40)
    print("FORECASTING RESULTS")
    print("="*40)
    
    print(f"\nBand decomposition types: {band_types}")
    
    print(f"\nRMSE Comparison:")
    print(f"Baseline GARCH      : {rmse(test, base_fcst):.6f}")
    print(f"Simple ELM          : {rmse(test, elm_fcst):.6f}")
    print(f"Enhanced ABS-VM     : {rmse(test, enhanced_forecast):.6f}")
    
    print(f"\nMAE Comparison:")
    print(f"Baseline GARCH      : {mean_absolute_error(test, base_fcst):.6f}")
    print(f"Simple ELM          : {mean_absolute_error(test, elm_fcst):.6f}")
    print(f"Enhanced ABS-VM     : {mean_absolute_error(test, enhanced_forecast):.6f}")
    
    # Directional accuracy
    def directional_accuracy(y_true, y_pred):
        correct = ((y_true[1:] > y_true[:-1]) == (y_pred[1:] > y_pred[:-1])).sum()
        return correct / (len(y_true) - 1)
    
    print(f"\nDirectional Accuracy:")
    print(f"Baseline GARCH      : {directional_accuracy(test, base_fcst):.2%}")
    print(f"Simple ELM          : {directional_accuracy(test, elm_fcst):.2%}")
    print(f"Enhanced ABS-VM     : {directional_accuracy(test, enhanced_forecast):.2%}")


if __name__ == "__main__":
    main()