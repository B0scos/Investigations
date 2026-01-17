import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from arch import arch_model
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")  # optional, keeps output clean

# ---------------------------
# 2. Compute daily realized variance
# ---------------------------
def compute_daily_rv(df):
    df = df.copy()
    df['log_return'] = np.log(df['close']).diff()
    df['date'] = df['timestamp'].dt.date
    
    # Handle potential NaN values
    df['log_return'] = df['log_return'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Debug: Check for zeros or very small values
    if df['log_return'].abs().max() < 1e-10:
        print("Warning: Very small or zero log returns detected")
    
    rv = df.groupby('date')['log_return'].apply(lambda x: np.sum(x**2))
    
    # Debug output
    print(f"RV computed for {len(rv)} days")
    print(f"RV range: [{rv.min():.8f}, {rv.max():.8f}]")
    
    return rv

# ---------------------------
# 3. Prepare features
# ---------------------------
def prepare_features(rv):
    # Convert to DataFrame if Series
    if isinstance(rv, pd.Series):
        df = pd.DataFrame({'RV_d': rv})
    else:
        df = rv.copy()
    
    # Check for zeros or NaNs
    if df['RV_d'].isnull().any():
        print("Warning: NaN values in RV_d, filling with forward fill")
        df['RV_d'] = df['RV_d'].fillna(method='ffill').fillna(method='bfill')
    
    # Replace zeros with small value to avoid issues with logs or divisions
    zero_mask = df['RV_d'] == 0
    if zero_mask.any():
        print(f"Warning: {zero_mask.sum()} zero values in RV_d, replacing with 1e-10")
        df.loc[zero_mask, 'RV_d'] = 1e-10
    
    # Calculate rolling features
    df['RV_w'] = df['RV_d'].rolling(7, min_periods=1).mean()
    df['RV_m'] = df['RV_d'].rolling(30, min_periods=1).mean()
    
    # Add lags
    df['RV_d_lag1'] = df['RV_d'].shift(1)
    df['RV_w_lag1'] = df['RV_w'].shift(1)
    df['RV_m_lag1'] = df['RV_m'].shift(1)
    
    # Fill initial NaN values from rolling means
    df = df.fillna(method='bfill')
    
    # Final check for any remaining NaNs
    if df.isnull().any().any():
        print("Warning: NaN values after feature preparation")
        df = df.dropna()
    
    X = df[['RV_d_lag1', 'RV_w_lag1', 'RV_m_lag1']].astype(float)
    y = df['RV_d'].astype(float)
    
    print(f"Features prepared: X shape = {X.shape}, y shape = {y.shape}")
    return X, y

# ---------------------------
# 4. Base model forecasts
# ---------------------------
def fit_garch(y_train):
    """GARCH(1,1) forecast with robust fallback"""
    # Ensure y_train is a numpy array and not constant
    y_train_series = pd.Series(y_train).dropna()
    
    if len(y_train_series) < 10:
        return float(np.var(y_train_series)) if len(y_train_series) > 0 else 1e-8
    
    # Check if series is constant
    if np.allclose(y_train_series, y_train_series.iloc[0]):
        return float(np.var(y_train_series))
    
    try:
        # Scale data for better numerical stability
        scale_factor = np.mean(np.abs(y_train_series))
        if scale_factor < 1e-10:
            scale_factor = 1.0
            
        y_scaled = y_train_series / scale_factor
        
        am = arch_model(y_scaled, vol='Garch', p=1, q=1, dist='normal', 
                       mean='Zero', rescale=False)
        res = am.fit(disp='off', show_warning=False, options={'maxiter': 1000})
        
        forecast = res.forecast(horizon=1).variance.values[-1, 0] * scale_factor
        
        return float(forecast)
        
    except Exception as e:
        # Fallback to variance
        fallback = float(np.var(y_train_series))
        return fallback if not np.isnan(fallback) else 1e-8

def base_forecasts(X_train, y_train, X_pred):
    """Generate point forecasts for Ridge, HAR, GARCH"""
    # Ensure data is properly formatted
    X_train = pd.DataFrame(X_train).reset_index(drop=True).astype(float)
    y_train = pd.Series(y_train).reset_index(drop=True).astype(float)
    X_pred = pd.DataFrame(X_pred).reset_index(drop=True).astype(float)
    
    # Ridge regression with regularization
    ridge = Ridge(alpha=1.0, random_state=42).fit(X_train, y_train)
    y_hat_ridge = ridge.predict(X_pred)
    
    # HAR approximation - weighted average as in original HAR paper
    weights = np.array([0.5, 0.3, 0.2])  # Daily, weekly, monthly weights
    y_hat_har = (X_pred.values * weights).sum(axis=1)
    
    # GARCH(1,1) forecasts - one forecast per test sample
    y_hat_garch = np.array([fit_garch(y_train) for _ in range(len(X_pred))])
    
    return pd.DataFrame({
        'ridge': y_hat_ridge,
        'har': y_hat_har,
        'garch': y_hat_garch
    }, index=X_pred.index)

# ---------------------------
# 5. Quantile Forecasting Methods
# ---------------------------
def qrs(y_hat_train, y_train, y_hat_test, quantiles=[0.01, 0.05, 0.5, 0.95, 0.99]):
    """Quantile Regression Score (residual-based)"""
    # Calculate residuals from ensemble mean
    ensemble_mean = y_hat_train.mean(axis=1).values
    residuals = y_train.values - ensemble_mean
    
    # Remove NaN/inf values
    residuals = residuals[~np.isnan(residuals) & ~np.isinf(residuals)]
    
    # Generate predictions by adding residuals to test ensemble mean
    test_ensemble_mean = y_hat_test.mean(axis=1).values[:, None]
    simulated = test_ensemble_mean + residuals
    
    # Calculate quantiles
    quantile_preds = np.percentile(simulated, [q * 100 for q in quantiles], axis=1).T
    
    return pd.DataFrame(quantile_preds, columns=[f'q{int(q*100)}' for q in quantiles])

def qlr(y_hat_train, y_train, y_hat_test, quantiles=[0.01, 0.05, 0.5, 0.95, 0.99]):
    """Quantile Linear Regression"""
    y_train = pd.Series(y_train).reset_index(drop=True).astype(float)
    y_hat_train = pd.DataFrame(y_hat_train).reset_index(drop=True).astype(float)
    y_hat_test = pd.DataFrame(y_hat_test).reset_index(drop=True).astype(float)
    
    # Add constant term
    y_hat_train_const = y_hat_train.copy()
    y_hat_test_const = y_hat_test.copy()
    y_hat_train_const['const'] = 1
    y_hat_test_const['const'] = 1
    
    preds = []
    for q in quantiles:
        try:
            model = QuantReg(y_train, y_hat_train_const).fit(q=q, max_iter=5000)
            preds.append(model.predict(y_hat_test_const))
        except Exception as e:
            # Fallback to median of training data
            print(f"Warning: QLR failed for q={q}, using fallback")
            fallback_pred = np.percentile(y_train, q * 100)
            preds.append(np.full(len(y_hat_test), fallback_pred))
    
    return pd.DataFrame(np.column_stack(preds), columns=[f'q{int(q*100)}' for q in quantiles])

def qrf(y_hat_train, y_train, y_hat_test, quantiles=[0.01, 0.05, 0.5, 0.95, 0.99]):
    """Quantile Random Forest"""
    y_train = pd.Series(y_train).reset_index(drop=True).astype(float)
    y_hat_train = pd.DataFrame(y_hat_train).reset_index(drop=True).astype(float)
    y_hat_test = pd.DataFrame(y_hat_test).reset_index(drop=True).astype(float)
    
    # Remove any NaN values
    valid_idx = ~y_train.isna() & ~y_hat_train.isna().any(axis=1)
    y_train = y_train[valid_idx]
    y_hat_train = y_hat_train[valid_idx]
    
    # Use sklearn's QuantileRandomForest if available, otherwise implement manually
    try:
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42, 
                                  min_samples_leaf=5, n_jobs=-1)
        rf.fit(y_hat_train, y_train)
        
        # Get predictions from each tree
        all_tree_preds = []
        for tree in rf.estimators_:
            tree_pred = tree.predict(y_hat_test)
            all_tree_preds.append(tree_pred)
        
        all_tree_preds = np.array(all_tree_preds)  # shape: (n_trees, n_samples)
        
        # Calculate quantiles across trees
        quantile_preds = []
        for q in quantiles:
            q_pred = np.percentile(all_tree_preds, q * 100, axis=0)
            quantile_preds.append(q_pred)
            
    except Exception as e:
        print(f"Warning: QRF failed, using simple fallback: {e}")
        # Simple fallback using training distribution
        quantile_preds = []
        for q in quantiles:
            q_val = np.percentile(y_train, q * 100)
            quantile_preds.append(np.full(len(y_hat_test), q_val))
    
    return pd.DataFrame(np.column_stack(quantile_preds), 
                       columns=[f'q{int(q*100)}' for q in quantiles])

def direct_qr(X_train, y_train, X_test, quantiles=[0.01, 0.05, 0.5, 0.95, 0.99]):
    """Direct quantile regression using lagged features"""
    # Reset indices
    y_train = pd.Series(y_train).reset_index(drop=True).astype(float)
    X_train = pd.DataFrame(X_train).reset_index(drop=True).astype(float)
    X_test = pd.DataFrame(X_test).reset_index(drop=True).astype(float)
    
    # Add constant term
    X_train_const = X_train.copy()
    X_test_const = X_test.copy()
    X_train_const['const'] = 1
    X_test_const['const'] = 1
    
    preds = []
    for q in quantiles:
        try:
            model = QuantReg(y_train, X_train_const).fit(q=q, max_iter=5000)
            pred = model.predict(X_test_const)
            preds.append(pred)
        except Exception as e:
            print(f"Warning: Direct QR failed for q={q}, using fallback: {e}")
            # Use empirical quantile as fallback
            q_val = np.percentile(y_train, q * 100)
            preds.append(np.full(len(X_test), q_val))
    
    return pd.DataFrame(np.column_stack(preds), 
                       columns=[f'dqr_q{int(q*100)}' for q in quantiles])

# ---------------------------
# 6. Quantile Regression Metrics
# ---------------------------
def pinball_loss(y_true, y_pred_quantile, q):
    """Calculate pinball loss for a specific quantile"""
    error = y_true - y_pred_quantile
    loss = np.where(error >= 0, q * error, (q - 1) * error)
    return np.mean(loss)

def mean_pinball_loss(y_true, quantile_preds, quantiles):
    """Calculate mean pinball loss across all quantiles"""
    total_loss = 0
    for i, q in enumerate(quantiles):
        total_loss += pinball_loss(y_true, quantile_preds.iloc[:, i].values, q)
    return total_loss / len(quantiles)

def winkler_score(y_true, lower_q, upper_q, alpha=0.1):
    """Winkler Score for prediction intervals"""
    interval_width = upper_q - lower_q
    score = interval_width.copy()
    
    # Penalize if below lower bound
    below_lower = y_true < lower_q
    score[below_lower] += (2/alpha) * (lower_q[below_lower] - y_true[below_lower])
    
    # Penalize if above upper bound
    above_upper = y_true > upper_q
    score[above_upper] += (2/alpha) * (y_true[above_upper] - upper_q[above_upper])
    
    return np.mean(score)

def calculate_interval_coverage(y_true, lower_q, upper_q):
    """Calculate percentage of true values within prediction interval"""
    within_interval = (y_true >= lower_q) & (y_true <= upper_q)
    return np.mean(within_interval) * 100

def calculate_quantile_crossing(quantile_preds, quantiles):
    """Calculate percentage of quantile crossing cases"""
    if len(quantile_preds) == 0:
        return 0
    
    quantile_crossing = 0
    for i in range(len(quantile_preds)):
        for j in range(1, len(quantiles)):
            if quantile_preds.iloc[i, j] < quantile_preds.iloc[i, j-1]:
                quantile_crossing += 1
                break
    return (quantile_crossing / len(quantile_preds)) * 100

def collect_model_metrics(y_test, model_name, point_preds=None, quantile_preds=None, quantiles=None):
    """Collect all metrics for a model"""
    metrics = {'Model': model_name}
    
    # Point forecast metrics
    if point_preds is not None:
        if len(y_test) == len(point_preds):
            metrics['RMSE'] = np.sqrt(mean_squared_error(y_test, point_preds))
            metrics['MAE'] = mean_absolute_error(y_test, point_preds)
        else:
            print(f"Warning: Length mismatch for {model_name}: y_test={len(y_test)}, preds={len(point_preds)}")
    
    # Quantile forecast metrics
    if quantile_preds is not None and quantiles is not None:
        if len(y_test) != len(quantile_preds):
            print(f"Warning: Length mismatch in quantile predictions for {model_name}")
            return metrics
        
        # Find median index
        median_idx = quantiles.index(0.5) if 0.5 in quantiles else None
        
        # Median point forecast metrics
        if median_idx is not None:
            y_pred_median = quantile_preds.iloc[:, median_idx].values
            if len(y_test) == len(y_pred_median):
                metrics['Median_RMSE'] = np.sqrt(mean_squared_error(y_test, y_pred_median))
                metrics['Median_MAE'] = mean_absolute_error(y_test, y_pred_median)
        
        # Pinball losses
        try:
            metrics['Mean_Pinball_Loss'] = mean_pinball_loss(y_test.values, quantile_preds, quantiles)
        except:
            metrics['Mean_Pinball_Loss'] = np.nan
        
        # Individual quantile pinball losses
        for i, q in enumerate(quantiles):
            try:
                metrics[f'Pinball_q{int(q*100)}'] = pinball_loss(
                    y_test.values, quantile_preds.iloc[:, i].values, q
                )
            except:
                metrics[f'Pinball_q{int(q*100)}'] = np.nan
        
        # Prediction interval metrics (90% PI)
        if 0.05 in quantiles and 0.95 in quantiles:
            q05_idx = quantiles.index(0.05)
            q95_idx = quantiles.index(0.95)
            lower = quantile_preds.iloc[:, q05_idx].values
            upper = quantile_preds.iloc[:, q95_idx].values
            
            metrics['Winkler_Score'] = winkler_score(y_test.values, lower, upper, alpha=0.1)
            metrics['PI_Coverage_90%'] = calculate_interval_coverage(y_test.values, lower, upper)
            metrics['PI_Width_Avg'] = np.mean(upper - lower)
        
        # Quantile crossing
        metrics['Quantile_Crossing_%'] = calculate_quantile_crossing(quantile_preds, quantiles)
        
        # Direction win rate (using median as point forecast)
        if median_idx is not None and len(y_test) > 1:
            y_pred_median = quantile_preds.iloc[:, median_idx].values
            if len(y_pred_median) == len(y_test):
                actual_directions = (y_test.values[1:] > y_test.values[:-1]).astype(int)
                predicted_directions = (y_pred_median[1:] > y_test.values[:-1]).astype(int)
                
                if len(actual_directions) == len(predicted_directions):
                    correct_predictions = (predicted_directions == actual_directions)
                    metrics['Direction_Win_Rate_%'] = np.mean(correct_predictions) * 100
                else:
                    metrics['Direction_Win_Rate_%'] = np.nan
            else:
                metrics['Direction_Win_Rate_%'] = np.nan
    
    return metrics

# ---------------------------
# 7. Simple Direction Win Rate
# ---------------------------
def calculate_direction_win_rate(y_true, y_pred, model_name=""):
    """Simple win rate for volatility direction prediction"""
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0, 0, 0
    
    # Ensure lengths match
    min_len = min(len(y_true) - 1, len(y_pred) - 1)
    
    actual_directions = (y_true.values[1:min_len+1] > y_true.values[:min_len]).astype(int)
    predicted_directions = (y_pred[1:min_len+1] > y_true.values[:min_len]).astype(int)
    
    correct_predictions = (predicted_directions == actual_directions)
    win_rate = np.mean(correct_predictions) * 100 if len(correct_predictions) > 0 else 0
    
    return win_rate, len(correct_predictions), np.sum(correct_predictions)

# ---------------------------
# 8. Main pipeline
# ---------------------------
if __name__ == "__main__":
    try:
        # Load data
        print("Loading data...")
        df = pd.read_csv('data/BTC_data_hour.csv')
        
        # Check required columns
        required_cols = ['timestamp', 'close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Data loaded: {len(df)} rows, columns: {df.columns.tolist()}")
        
        print("\nComputing realized variance...")
        rv = compute_daily_rv(df)
        
        print("\nPreparing features...")
        X, y = prepare_features(rv)
        
        if len(X) < 50:
            raise ValueError(f"Not enough data to train models. Samples: {len(X)}")
        
        # Split data (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )
        
        print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        # Generate base model forecasts
        print("\nGenerating base model forecasts...")
        y_hat_train = base_forecasts(X_train, y_train, X_train)
        y_hat_test = base_forecasts(X_train, y_train, X_test)
        
        print(f"Base forecasts generated: {y_hat_test.shape}")
        
        # Generate probabilistic forecasts
        print("\nGenerating probabilistic forecasts...")
        quantile_levels = [0.01, 0.05, 0.5, 0.95, 0.99]
        
        qrs_preds = qrs(y_hat_train, y_train, y_hat_test, quantile_levels)
        qlr_preds = qlr(y_hat_train, y_train, y_hat_test, quantile_levels)
        qrf_preds = qrf(y_hat_train, y_train, y_hat_test, quantile_levels)
        
        # Generate direct quantile regression forecasts
        print("\nGenerating DIRECT quantile regression forecasts...")
        dqr_preds = direct_qr(X_train, y_train, X_test, quantile_levels)
        
        # ================================================
        # Collect all metrics into DataFrame
        # ================================================
        print("\n" + "="*60)
        print("COLLECTING ALL METRICS")
        print("="*60)
        
        all_metrics = []
        
        # Collect metrics for base models
        for model_name in y_hat_test.columns:
            metrics = collect_model_metrics(
                y_test=y_test,
                model_name=f"BASE_{model_name.upper()}",
                point_preds=y_hat_test[model_name].values
            )
            all_metrics.append(metrics)
            print(f"  Collected metrics for {model_name}")
        
        # Collect metrics for QRS
        if qrs_preds is not None and len(qrs_preds) > 0:
            metrics = collect_model_metrics(
                y_test=y_test,
                model_name="QRS",
                quantile_preds=qrs_preds,
                quantiles=quantile_levels
            )
            all_metrics.append(metrics)
            print("  Collected metrics for QRS")
        
        # Collect metrics for QLR
        if qlr_preds is not None and len(qlr_preds) > 0:
            metrics = collect_model_metrics(
                y_test=y_test,
                model_name="QLR",
                quantile_preds=qlr_preds,
                quantiles=quantile_levels
            )
            all_metrics.append(metrics)
            print("  Collected metrics for QLR")
        
        # Collect metrics for QRF
        if qrf_preds is not None and len(qrf_preds) > 0:
            metrics = collect_model_metrics(
                y_test=y_test,
                model_name="QRF",
                quantile_preds=qrf_preds,
                quantiles=quantile_levels
            )
            all_metrics.append(metrics)
            print("  Collected metrics for QRF")
        
        # Collect metrics for Direct QR
        if dqr_preds is not None and len(dqr_preds) > 0:
            metrics = collect_model_metrics(
                y_test=y_test,
                model_name="DIRECT_QR",
                quantile_preds=dqr_preds,
                quantiles=quantile_levels
            )
            all_metrics.append(metrics)
            print("  Collected metrics for DIRECT_QR")
        
        # Create DataFrame from all metrics
        results_df = pd.DataFrame(all_metrics)
        
        # Display results
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        
        # Show key metrics
        key_cols = ['Model', 'RMSE', 'MAE', 'Median_RMSE', 'Mean_Pinball_Loss', 
                   'PI_Coverage_90%', 'Direction_Win_Rate_%']
        
        # Filter to available columns
        available_cols = [col for col in key_cols if col in results_df.columns]
        if available_cols:
            print(results_df[available_cols].to_string(index=False))
        else:
            print(results_df.to_string(index=False))
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"quantile_regression_results_{timestamp}.csv"
        results_df.to_csv(csv_filename, index=False)
        print(f"\nResults saved to: {csv_filename}")
        
        # Simple evaluation print
        print("\n" + "="*60)
        print("QUICK EVALUATION")
        print("="*60)
        
        if 'RMSE' in results_df.columns:
            best_rmse_idx = results_df['RMSE'].idxmin()
            print(f"Best RMSE: {results_df.loc[best_rmse_idx, 'Model']} = {results_df.loc[best_rmse_idx, 'RMSE']:.8f}")
        
        if 'Direction_Win_Rate_%' in results_df.columns:
            best_win_idx = results_df['Direction_Win_Rate_%'].idxmax()
            print(f"Best Win Rate: {results_df.loc[best_win_idx, 'Model']} = {results_df.loc[best_win_idx, 'Direction_Win_Rate_%']:.2f}%")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()