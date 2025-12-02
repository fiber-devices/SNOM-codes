"""
Nonlinear Phase Model Fitting for Michelson Interferometer
===========================================================
Build a nonlinear phase model from training data (40 samples),
then fit new MI signals using this model to minimize residuals.

Use case: 
- You have 40 calibration samples (time + MI signal)
- Want to learn the nonlinear phase relationship
- Apply this model to new MI signals by fitting parameters
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit, differential_evolution
from scipy.interpolate import UnivariateSpline, interp1d
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class NonlinearPhaseFitter:
    """
    Builds nonlinear phase model from training data and fits new signals.
    """
    
    def __init__(self):
        """Initialize the fitter."""
        self.phase_model = None
        self.model_type = None
        self.training_data = {}
        self.fit_results = []
        
    def load_training_data(self, training_samples):
        """
        Load 40 training samples.
        
        Parameters:
        -----------
        training_samples : list of dict
            List of 40 samples, each with {'time': array, 'mi_signal': array}
        """
        print("=" * 70)
        print("LOADING TRAINING DATA")
        print("=" * 70)
        
        self.training_samples = training_samples
        n_samples = len(training_samples)
        
        print(f"\nNumber of training samples: {n_samples}")
        
        # Extract statistics from each sample
        for i, sample in enumerate(training_samples):
            time = np.array(sample['time'])
            mi_signal = np.array(sample['mi_signal'])
            
            if i == 0:
                print(f"\nSample 0 info:")
                print(f"  Length: {len(time)} points")
                print(f"  Time range: {time[0]:.6f} to {time[-1]:.6f}")
                print(f"  MI range: {mi_signal.min():.6f} to {mi_signal.max():.6f}")
        
        self.training_data = {
            'samples': training_samples,
            'n_samples': n_samples
        }
        
        return self.training_data
    
    def build_phase_model(self, model_type='polynomial', degree=10):
        """
        Build nonlinear phase model from training data.
        
        Parameters:
        -----------
        model_type : str
            'polynomial', 'fourier', 'spline', 'ml' (machine learning)
        degree : int
            Degree for polynomial or number of Fourier terms
        """
        print("\n" + "=" * 70)
        print(f"BUILDING NONLINEAR PHASE MODEL: {model_type.upper()}")
        print("=" * 70)
        
        # Extract phase from all training samples
        all_mi_signals = []
        all_phases = []
        
        for i, sample in enumerate(self.training_samples):
            mi_signal = np.array(sample['mi_signal'])
            time = np.array(sample['time'])
            
            # Extract phase using Hilbert transform
            from scipy.signal import hilbert
            analytic_signal = hilbert(mi_signal - np.mean(mi_signal))
            phase = np.unwrap(np.angle(analytic_signal))
            
            # Normalize MI signal to [0, 1]
            mi_norm = (mi_signal - mi_signal.min()) / (mi_signal.max() - mi_signal.min())
            
            all_mi_signals.extend(mi_norm)
            all_phases.extend(phase)
        
        all_mi_signals = np.array(all_mi_signals)
        all_phases = np.array(all_phases)
        
        print(f"\nTotal data points: {len(all_mi_signals)}")
        print(f"MI signal range: {all_mi_signals.min():.6f} to {all_mi_signals.max():.6f}")
        print(f"Phase range: {all_phases.min():.6f} to {all_phases.max():.6f} rad")
        
        if model_type == 'polynomial':
            # Polynomial regression: phase = f(MI_signal)
            print(f"\nFitting polynomial of degree {degree}...")
            
            # Create polynomial features
            poly = PolynomialFeatures(degree=degree, include_bias=True)
            X_poly = poly.fit_transform(all_mi_signals.reshape(-1, 1))
            
            # Fit with regularization
            model = Ridge(alpha=1.0)
            model.fit(X_poly, all_phases)
            
            # Store model
            self.phase_model = {
                'type': 'polynomial',
                'poly_transform': poly,
                'regressor': model,
                'degree': degree
            }
            
            # Evaluate fit quality
            phase_pred = model.predict(X_poly)
            residuals = all_phases - phase_pred
            rmse = np.sqrt(np.mean(residuals**2))
            
            print(f"  RMSE: {rmse:.6f} rad")
            print(f"  Max residual: {np.max(np.abs(residuals)):.6f} rad")
            
        elif model_type == 'fourier':
            # Fourier series: phase = a0 + sum(an*cos(n*mi) + bn*sin(n*mi))
            print(f"\nFitting Fourier series with {degree} terms...")
            
            def fourier_model(mi, *params):
                a0 = params[0]
                result = np.full_like(mi, a0)
                for n in range(1, degree + 1):
                    an = params[2*n - 1]
                    bn = params[2*n]
                    result += an * np.cos(2 * np.pi * n * mi)
                    result += bn * np.sin(2 * np.pi * n * mi)
                return result
            
            # Initial guess
            p0 = [np.mean(all_phases)] + [0.1] * (2 * degree)
            
            # Fit
            try:
                popt, _ = curve_fit(fourier_model, all_mi_signals, all_phases, 
                                   p0=p0, maxfev=10000)
                
                self.phase_model = {
                    'type': 'fourier',
                    'coefficients': popt,
                    'n_terms': degree,
                    'model_func': fourier_model
                }
                
                phase_pred = fourier_model(all_mi_signals, *popt)
                residuals = all_phases - phase_pred
                rmse = np.sqrt(np.mean(residuals**2))
                
                print(f"  RMSE: {rmse:.6f} rad")
                
            except Exception as e:
                print(f"  Warning: Fourier fit failed ({e}), falling back to polynomial")
                return self.build_phase_model('polynomial', degree)
        
        elif model_type == 'spline':
            # Spline interpolation
            print(f"\nFitting spline with smoothing parameter {degree}...")
            
            # Sort by MI signal for spline fitting
            sort_idx = np.argsort(all_mi_signals)
            mi_sorted = all_mi_signals[sort_idx]
            phase_sorted = all_phases[sort_idx]
            
            # Create spline (degree parameter used as smoothing factor)
            spline = UnivariateSpline(mi_sorted, phase_sorted, s=degree, k=3)
            
            self.phase_model = {
                'type': 'spline',
                'spline': spline
            }
            
            phase_pred = spline(all_mi_signals)
            residuals = all_phases - phase_pred
            rmse = np.sqrt(np.mean(residuals**2))
            
            print(f"  RMSE: {rmse:.6f} rad")
        
        elif model_type == 'ml':
            # Machine learning approach (Random Forest)
            print(f"\nTraining Random Forest with {degree} estimators...")
            
            model = RandomForestRegressor(
                n_estimators=degree,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            
            model.fit(all_mi_signals.reshape(-1, 1), all_phases)
            
            self.phase_model = {
                'type': 'ml',
                'model': model
            }
            
            phase_pred = model.predict(all_mi_signals.reshape(-1, 1))
            residuals = all_phases - phase_pred
            rmse = np.sqrt(np.mean(residuals**2))
            
            print(f"  RMSE: {rmse:.6f} rad")
        
        self.model_type = model_type
        print(f"\n✓ Nonlinear phase model built successfully!")
        
        return self.phase_model
    
    def predict_phase(self, mi_signal):
        """
        Predict phase from MI signal using trained model.
        
        Parameters:
        -----------
        mi_signal : array
            MI signal values
        
        Returns:
        --------
        phase : array
            Predicted phase values
        """
        if self.phase_model is None:
            raise ValueError("No model trained. Call build_phase_model() first.")
        
        # Normalize MI signal
        mi_norm = (mi_signal - mi_signal.min()) / (mi_signal.max() - mi_signal.min())
        
        if self.phase_model['type'] == 'polynomial':
            X_poly = self.phase_model['poly_transform'].transform(mi_norm.reshape(-1, 1))
            phase = self.phase_model['regressor'].predict(X_poly)
            
        elif self.phase_model['type'] == 'fourier':
            phase = self.phase_model['model_func'](mi_norm, *self.phase_model['coefficients'])
            
        elif self.phase_model['type'] == 'spline':
            phase = self.phase_model['spline'](mi_norm)
            
        elif self.phase_model['type'] == 'ml':
            phase = self.phase_model['model'].predict(mi_norm.reshape(-1, 1))
        
        return phase
    
    def fit_new_signal(self, time, mi_signal, wavelength=632.8e-9, method='optimize'):
        """
        Fit new MI signal using the nonlinear phase model.
        
        Parameters:
        -----------
        time : array
            Time values
        mi_signal : array
            New MI signal to fit
        wavelength : float
            Laser wavelength in meters
        method : str
            'optimize' or 'direct'
        
        Returns:
        --------
        fit_result : dict
            Fitting results including parameters and residuals
        """
        print("\n" + "=" * 70)
        print("FITTING NEW MI SIGNAL")
        print("=" * 70)
        
        time = np.array(time)
        mi_signal = np.array(mi_signal)
        
        print(f"\nSignal info:")
        print(f"  Length: {len(time)} points")
        print(f"  Time range: {time[0]:.6f} to {time[-1]:.6f}")
        print(f"  MI range: {mi_signal.min():.6f} to {mi_signal.max():.6f}")
        
        if method == 'direct':
            # Direct approach: predict phase from model
            predicted_phase = self.predict_phase(mi_signal)
            
            # Reconstruct signal from phase
            # MI = I0 * [1 + V * cos(phase)]
            def signal_model(phase, I0, V, offset):
                return I0 * (1 + V * np.cos(phase + offset))
            
            # Fit amplitude and visibility parameters
            def objective(params):
                I0, V, offset = params
                mi_pred = signal_model(predicted_phase, I0, V, offset)
                return np.sum((mi_signal - mi_pred)**2)
            
            # Initial guess
            x0 = [np.mean(mi_signal), 0.9, 0.0]
            bounds = [(0, 2*np.max(mi_signal)), (0, 1), (-2*np.pi, 2*np.pi)]
            
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
            
            I0_fit, V_fit, offset_fit = result.x
            mi_fitted = signal_model(predicted_phase, I0_fit, V_fit, offset_fit)
            residuals = mi_signal - mi_fitted
            
            fit_result = {
                'method': 'direct',
                'I0': I0_fit,
                'visibility': V_fit,
                'phase_offset': offset_fit,
                'phase': predicted_phase,
                'fitted_signal': mi_fitted,
                'residuals': residuals,
                'rmse': np.sqrt(np.mean(residuals**2)),
                'r_squared': 1 - np.sum(residuals**2) / np.sum((mi_signal - np.mean(mi_signal))**2)
            }
            
        elif method == 'optimize':
            # Optimization approach: optimize all parameters including phase modulation
            
            # Parameterized phase model
            def parameterized_phase(t, amplitude, frequency, phase_offset, *nonlinear_params):
                # Base sinusoidal movement
                base_phase = 2 * np.pi * frequency * t + phase_offset
                
                # Add nonlinear terms
                nonlinear = 0
                for i, param in enumerate(nonlinear_params):
                    nonlinear += param * np.sin((i+2) * base_phase)
                
                return amplitude * (base_phase + nonlinear)
            
            def full_model(t, I0, V, amplitude, frequency, phase_offset, *nonlinear_params):
                phase = parameterized_phase(t, amplitude, frequency, phase_offset, *nonlinear_params)
                return I0 * (1 + V * np.cos(phase))
            
            # Initial guess for parameters
            n_nonlinear = 3  # Number of nonlinear terms
            p0 = [
                np.mean(mi_signal),  # I0
                0.9,  # V
                1.0,  # amplitude
                1.0,  # frequency
                0.0,  # phase_offset
            ] + [0.0] * n_nonlinear  # nonlinear terms
            
            try:
                popt, pcov = curve_fit(full_model, time, mi_signal, p0=p0, maxfev=5000)
                
                I0_fit, V_fit, amp_fit, freq_fit, offset_fit = popt[:5]
                nonlinear_fit = popt[5:]
                
                mi_fitted = full_model(time, *popt)
                phase_fit = parameterized_phase(time, amp_fit, freq_fit, offset_fit, *nonlinear_fit)
                residuals = mi_signal - mi_fitted
                
                fit_result = {
                    'method': 'optimize',
                    'I0': I0_fit,
                    'visibility': V_fit,
                    'amplitude': amp_fit,
                    'frequency': freq_fit,
                    'phase_offset': offset_fit,
                    'nonlinear_params': nonlinear_fit,
                    'phase': phase_fit,
                    'fitted_signal': mi_fitted,
                    'residuals': residuals,
                    'rmse': np.sqrt(np.mean(residuals**2)),
                    'r_squared': 1 - np.sum(residuals**2) / np.sum((mi_signal - np.mean(mi_signal))**2)
                }
                
            except Exception as e:
                print(f"  Optimization failed: {e}")
                print("  Falling back to direct method...")
                return self.fit_new_signal(time, mi_signal, wavelength, method='direct')
        
        print(f"\nFitting results:")
        print(f"  Method: {fit_result['method']}")
        print(f"  I0 (mean intensity): {fit_result['I0']:.6f}")
        print(f"  Visibility: {fit_result['visibility']:.6f}")
        print(f"  RMSE: {fit_result['rmse']:.6f}")
        print(f"  R²: {fit_result['r_squared']:.6f}")
        print(f"  Max residual: {np.max(np.abs(residuals)):.6f}")
        
        self.fit_results.append(fit_result)
        return fit_result
    
    def plot_training_data(self, save_path='./DATA/training_data.png'):
        """Visualize training data."""
        n_samples = min(9, len(self.training_samples))
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for i in range(n_samples):
            sample = self.training_samples[i]
            time = np.array(sample['time'])
            mi_signal = np.array(sample['mi_signal'])
            
            axes[i].plot(time, mi_signal, 'b-', linewidth=1, alpha=0.7)
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('MI Signal')
            axes[i].set_title(f'Training Sample {i+1}')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_samples, 9):
            axes[i].axis('off')
        
        plt.tight_layout()
        # Ensure the directory for the save path exists, create it if it doesn't
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining data plot saved: {save_path}")
        
        return fig
    
    def plot_phase_model(self, save_path='./DATA/phase_model.png'):
        """Visualize the learned phase model."""
        if self.phase_model is None:
            print("No model to plot. Build model first.")
            return
        
        # Generate test points
        mi_test = np.linspace(0, 1, 1000)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Phase model
        ax = axes[0, 0]
        
        # Get phase predictions
        if self.phase_model['type'] == 'polynomial':
            X_poly = self.phase_model['poly_transform'].transform(mi_test.reshape(-1, 1))
            phase_pred = self.phase_model['regressor'].predict(X_poly)
        elif self.phase_model['type'] == 'fourier':
            phase_pred = self.phase_model['model_func'](mi_test, *self.phase_model['coefficients'])
        elif self.phase_model['type'] == 'spline':
            phase_pred = self.phase_model['spline'](mi_test)
        elif self.phase_model['type'] == 'ml':
            phase_pred = self.phase_model['model'].predict(mi_test.reshape(-1, 1))
        
        ax.plot(mi_test, phase_pred, 'b-', linewidth=2, label='Model')
        ax.set_xlabel('Normalized MI Signal')
        ax.set_ylabel('Phase (rad)')
        ax.set_title(f'Learned Nonlinear Phase Model ({self.model_type})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. Reconstructed MI signal
        ax = axes[0, 1]
        mi_reconstructed = 1 + 0.9 * np.cos(phase_pred)
        mi_reconstructed = (mi_reconstructed - mi_reconstructed.min()) / np.ptp(mi_reconstructed)
        
        ax.plot(mi_test, mi_reconstructed, 'r-', linewidth=2, label='Reconstructed')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Identity')
        ax.set_xlabel('Original MI Signal')
        ax.set_ylabel('Reconstructed MI Signal')
        ax.set_title('Signal Reconstruction Quality')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 3. Phase derivative (nonlinearity)
        ax = axes[1, 0]
        phase_derivative = np.gradient(phase_pred, mi_test)
        ax.plot(mi_test, phase_derivative, 'g-', linewidth=2)
        ax.set_xlabel('Normalized MI Signal')
        ax.set_ylabel('dφ/dI')
        ax.set_title('Phase Nonlinearity (Derivative)')
        ax.grid(True, alpha=0.3)
        
        # 4. Curvature (second derivative)
        ax = axes[1, 1]
        phase_curvature = np.gradient(phase_derivative, mi_test)
        ax.plot(mi_test, phase_curvature, 'm-', linewidth=2)
        ax.set_xlabel('Normalized MI Signal')
        ax.set_ylabel('d²φ/dI²')
        ax.set_title('Phase Curvature')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Ensure the directory for the save path exists, create it if it doesn't
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Phase model plot saved: {save_path}")
        
        return fig
    
    def plot_fit_result(self, fit_result, time, mi_signal, 
                       save_path='./DATA/fit_result.png'):
        """Visualize fitting results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Fitted signal
        ax = axes[0, 0]
        ax.plot(time, mi_signal, 'b-', linewidth=2, alpha=0.7, label='Original')
        ax.plot(time, fit_result['fitted_signal'], 'r--', linewidth=2, alpha=0.7, label='Fitted')
        ax.set_xlabel('Time')
        ax.set_ylabel('MI Signal')
        ax.set_title(f'Signal Fit (R² = {fit_result["r_squared"]:.6f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Residuals vs time
        ax = axes[0, 1]
        ax.plot(time, fit_result['residuals'], 'g-', linewidth=1, alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.fill_between(time, fit_result['residuals'], alpha=0.3)
        ax.set_xlabel('Time')
        ax.set_ylabel('Residuals')
        ax.set_title(f'Residuals (RMSE = {fit_result["rmse"]:.6f})')
        ax.grid(True, alpha=0.3)
        
        # 3. Actual vs fitted
        ax = axes[1, 0]
        ax.scatter(mi_signal, fit_result['fitted_signal'], alpha=0.5, s=20)
        min_val = min(mi_signal.min(), fit_result['fitted_signal'].min())
        max_val = max(mi_signal.max(), fit_result['fitted_signal'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax.set_xlabel('Actual MI Signal')
        ax.set_ylabel('Fitted MI Signal')
        ax.set_title('Actual vs Fitted')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # 4. Residual histogram
        ax = axes[1, 1]
        ax.hist(fit_result['residuals'], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Residual Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Residual Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Ensure the directory for the save path exists, create it if it doesn't
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Fit result plot saved: {save_path}")
        
        return fig


def example_usage():
    """Example with 40 synthetic training samples."""
    print("\n" + "=" * 70)
    print("EXAMPLE: NONLINEAR PHASE MODEL WITH 40 TRAINING SAMPLES")
    print("=" * 70)
    
    # Generate 40 training samples with different conditions
    np.random.seed(42)
    training_samples = []
    
    wavelength = 632.8e-9
    
    for i in range(40):
        # Each sample has different piezo amplitude and frequency
        n_points = np.random.randint(100, 300)
        time = np.linspace(0, np.random.uniform(1, 3), n_points)
        
        # Piezo movement with nonlinearity
        amplitude = np.random.uniform(0.5e-6, 3e-6)
        frequency = np.random.uniform(0.5, 3.0)
        piezo = amplitude * np.sin(2 * np.pi * frequency * time)
        
        # Add nonlinearity to piezo movement
        piezo += 0.1 * amplitude * np.sin(4 * np.pi * frequency * time)
        piezo += 0.05 * amplitude * np.sin(6 * np.pi * frequency * time)
        
        # Generate MI signal
        phase = 4 * np.pi * piezo / wavelength
        mi_signal = 1 + 0.9 * np.cos(phase)
        
        # Add noise
        mi_signal += 0.02 * np.random.randn(n_points)
        
        training_samples.append({
            'time': time,
            'mi_signal': mi_signal
        })
    
    print(f"\nGenerated {len(training_samples)} training samples")
    
    # Create fitter
    fitter = NonlinearPhaseFitter()
    
    # Load training data
    fitter.load_training_data(training_samples)
    
    # Visualize training data
    fitter.plot_training_data()
    
    # Build phase model (try different types)
    print("\n" + "=" * 70)
    print("COMPARING DIFFERENT MODEL TYPES")
    print("=" * 70)
    
    for model_type in ['polynomial', 'fourier', 'spline', 'ml']:
        print(f"\n--- {model_type.upper()} ---")
        fitter.build_phase_model(model_type=model_type, degree=5)
    
    # Use polynomial for final model
    fitter.build_phase_model(model_type='polynomial', degree=14)
    fitter.plot_phase_model()
    
    # Generate new test signal
    print("\n" + "=" * 70)
    print("TESTING ON NEW SIGNAL")
    print("=" * 70)
    
    time_new = np.linspace(0, 2, 200)
    piezo_new = 2e-6 * np.sin(2 * np.pi * 1.5 * time_new)
    piezo_new += 0.15 * 2e-6 * np.sin(4 * np.pi * 1.5 * time_new)
    phase_new = 4 * np.pi * piezo_new / wavelength
    mi_new = 1 + 0.85 * np.cos(phase_new)
    mi_new += 0.03 * np.random.randn(len(time_new))
    
    # Fit new signal
    fit_result = fitter.fit_new_signal(time_new, mi_new, method='optimize')
    fitter.plot_fit_result(fit_result, time_new, mi_new)
    
    return fitter


if __name__ == "__main__":
    fitter = example_usage()
    
    print("\n" + "=" * 70)
    print("TO USE WITH YOUR 40 SAMPLES:")
    print("=" * 70)
    print("""
# 1. Load your 40 training samples
training_samples = []
for i in range(40):
    # Load each sample
    time, mi_signal = load_sample(i)  # Your loading function
    training_samples.append({
        'time': time,
        'mi_signal': mi_signal
    })

# 2. Create fitter and load data
fitter = NonlinearPhaseFitter()
fitter.load_training_data(training_samples)

# 3. Build nonlinear phase model
fitter.build_phase_model(model_type='polynomial', degree=7)
fitter.plot_phase_model()

# 4. Fit new MI signals
time_new, mi_signal_new = load_new_signal()  # Your new signal
fit_result = fitter.fit_new_signal(time_new, mi_signal_new, method='optimize')
fitter.plot_fit_result(fit_result, time_new, mi_signal_new)

# 5. Access results
print(f"RMSE: {fit_result['rmse']}")
print(f"R²: {fit_result['r_squared']}")
fitted_signal = fit_result['fitted_signal']
residuals = fit_result['residuals']
    """)
