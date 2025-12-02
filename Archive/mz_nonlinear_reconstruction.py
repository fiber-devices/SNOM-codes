import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, optimize, interpolate
from scipy.signal import find_peaks, hilbert
from scipy.interpolate import UnivariateSpline
import time

# class MZSimulator:
#     """Simulate Mach-Zehnder interferometer with nonlinear PZT scan"""
    
#     def __init__(self, n_points=2000, n_fringes=10, nonlinearity_strength=0.2, 
#                  phase_noise_std=0.01, amplitude_noise_std=0.01, t_end=1):
#         """
#         Parameters:
#         -----------
#         n_points : int
#             Number of data points
#         n_fringes : int
#             Number of fringes in the scan
#         nonlinearity_strength : float
#             Strength of PZT nonlinearity (0 = linear, 1 = strong nonlinear)
#         phase_noise_std : float
#             Standard deviation of phase noise (radians)
#         amplitude_noise_std : float
#             Standard deviation of amplitude noise
#         """
#         self.n_points = n_points
#         self.n_fringes = n_fringes
#         self.nonlinearity_strength = nonlinearity_strength
#         self.phase_noise_std = phase_noise_std
#         self.amplitude_noise_std = amplitude_noise_std
#         self.t_end = t_end
        
#     def generate_nonlinear_scan(self, nonlin_type='cubic'):
#         """Generate nonlinear z coordinate (actual PZT position)"""
#         t = np.linspace(0, self.t_end, self.n_points)  # Normalized time/voltage

#         # add random phase shift
#         phase_shift = np.random.randn() * 2 * np.pi
        
#         if nonlin_type == 'cubic':
#             # Cubic nonlinearity (common in PZTs)
#             z = t + self.nonlinearity_strength * (t**3 -t)
#         elif nonlin_type == 'hysteresis':
#             # Simplified hysteresis-like nonlinearity
#             z = t + self.nonlinearity_strength * np.sin(2*np.pi*t) * t
#         elif nonlin_type == 'exponential':
#             # Exponential-like creep
#             z = t + self.nonlinearity_strength * (1 - np.exp(-5*t))/5 
#         else:
#             z = t  # Linear
            
#         # Normalize to get desired number of fringes
#         z = z * 2 * np.pi * self.n_fringes
#         return t, z
    
#     def generate_mz_signal(self, z):
#         """Generate MZ interferometer output signal"""
#         # Add phase noise
#         phase_noise = np.random.randn(len(z)) * self.phase_noise_std
        
#         # MZ signal: I = I0 * (1 + V * cos(kz + noise))
#         # Using sin^2 formulation: I = I0 * sin^2(kz/2)
#         visibility = 0.95  # Typical visibility
#         offset = 0.5
#         amplitude = 0.45
        
#         # Add amplitude noise
#         amp_noise = 1 + np.random.randn(len(z)) * self.amplitude_noise_std
        
#         signal = offset + amplitude * visibility * np.sin(z/2 + phase_noise)**2
#         signal = signal * amp_noise
        
#         return signal

class MZSimulator:
    """Simulate Mach–Zehnder / Michelson interferometer with nonlinear scan"""

    def __init__(self, n_points=2000, n_fringes=10, nonlinearity_strength=0.2,
                 phase_noise_std=0.01, amplitude_noise_std=0.01, t_end=1.0):
        """
        Parameters
        ----------
        n_points : int
            Number of sampled data points.
        n_fringes : int
            Number of optical fringes in one full scan.
        nonlinearity_strength : float
            Strength of the PZT nonlinearity (0 = linear, 1 = strong nonlinear).
        phase_noise_std : float
            Standard deviation of random phase noise (radians).
        amplitude_noise_std : float
            Standard deviation of multiplicative amplitude noise.
        t_end : float
            Total scan duration (s). Shorter time = faster scan speed.
        """
        self.n_points = n_points
        self.n_fringes = n_fringes
        self.nonlinearity_strength = nonlinearity_strength
        self.phase_noise_std = phase_noise_std
        self.amplitude_noise_std = amplitude_noise_std
        self.t_end = t_end

    # -------------------------------
    def generate_nonlinear_scan(self, nonlin_type='cubic'):
        """Generate nonlinear optical path scan z(t) for fixed fringe count"""
        t = np.linspace(0, self.t_end, self.n_points)
        phase_shift = np.random.randn() * 2 * np.pi

        if nonlin_type == 'cubic':
            z = t + self.nonlinearity_strength * (t**3 - t)
        elif nonlin_type == 'hysteresis':
            z = t + self.nonlinearity_strength * np.sin(2 * np.pi * t) * t
        elif nonlin_type == 'exponential':
            z = t + self.nonlinearity_strength * (1 - np.exp(-5 * t)) / 5
        else:
            z = t

        # Normalize so that total phase covers exactly `n_fringes` cycles
        z = (z - z.min()) / (z.max() - z.min())
        z = z * 2 * np.pi * self.n_fringes + phase_shift

        return t, z

    # -------------------------------
    def generate_signal(self, nonlin_type='cubic'):
        """Generate Mach–Zehnder intensity signal with Gaussian envelope"""
        t, z = self.generate_nonlinear_scan(nonlin_type)

        envelope = 1

        # Random noise
        phase_noise = np.random.randn(len(t)) * self.phase_noise_std
        amp_noise = 1 + np.random.randn(len(t)) * self.amplitude_noise_std

        # Interference pattern
        signal = 0.5 + 0.45 * envelope * np.cos(z + phase_noise)
        signal *= amp_noise

        return t, signal

    # -------------------------------
    def plot_example(self):
        """Visualize how scan speed affects interference signal"""
        plt.figure(figsize=(7, 4))
        for t_end in [0.2, 0.5, 1.0]:
            self.t_end = t_end
            t, sig = self.generate_signal('cubic')
            plt.plot(t, sig, label=f't_end={t_end:.2f}s')
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized intensity")
        plt.legend()
        plt.grid(True)
        plt.title("Michelson / MZ Signal with Fixed Fringes, Varying Scan Speed")
        plt.tight_layout()
        plt.show()


class ZReconstructor:
    """Methods to reconstruct z coordinate from MZ signal"""
    
    def __init__(self, signal, sampling_points):
        """
        Parameters:
        -----------
        signal : array
            MZ interferometer signal
        sampling_points : array
            Sampling points (e.g., time or voltage)
        """
        self.signal = signal
        self.sampling_points = sampling_points
        self.n_points = len(signal)
        
    def method1_peak_counting(self):
        """Method 1: Simple peak counting with interpolation"""
        # Find peaks and valleys
        peaks, _ = find_peaks(self.signal, prominence=0.1)
        valleys, _ = find_peaks(-self.signal, prominence=0.1)
        
        # Combine and sort
        extrema = np.sort(np.concatenate([peaks, valleys]))
        
        if len(extrema) < 3:
            return np.zeros_like(self.signal)
        
        # Each extremum represents π/2 phase change
        extrema_phase = np.arange(len(extrema)) * np.pi/2
        
        # Interpolate to get phase at all points
        f = interpolate.interp1d(extrema, extrema_phase, kind='cubic', 
                                fill_value='extrapolate')
        z_reconstructed = f(np.arange(self.n_points))
        
        return z_reconstructed
    
    def method2_hilbert_transform(self):
        """Method 2: Hilbert transform for phase extraction"""
        # Remove DC component
        signal_ac = self.signal - np.mean(self.signal)
        
        # Apply Hilbert transform
        analytic_signal = hilbert(signal_ac)
        
        # Extract instantaneous phase
        phase = np.unwrap(np.angle(analytic_signal))
        
        # The signal is sin^2(z/2), which has frequency doubled
        # So we need to account for this
        z_reconstructed = 2*phase
        
        return z_reconstructed
    
    def method3_arccosine_method(self):
        """Method 3: Direct arccos inversion with unwrapping"""
        # Normalize signal to [0, 1]
        sig_min = np.min(self.signal)
        sig_max = np.max(self.signal)
        signal_norm = (self.signal - sig_min) / (sig_max - sig_min)
        
        # For sin^2(z/2), we have: signal = sin^2(z/2)
        # So: z = 2 * arcsin(sqrt(signal))
        # Handle numerical errors
        signal_norm = np.clip(signal_norm, 0, 1)
        
        # Calculate phase
        phase = 2 * np.arcsin(np.sqrt(signal_norm))
        
        # Unwrap phase by detecting jumps
        z_reconstructed = self._smart_unwrap(phase)
        
        return z_reconstructed
    
    def method4_fringe_tracking(self):
        """Method 4: Advanced fringe tracking with local frequency estimation"""
        # Find zero crossings of derivative
        sig_smooth = signal.savgol_filter(self.signal, 11, 3)
        sig_deriv = np.gradient(sig_smooth)
        
        # Find zero crossings
        zero_crossings = np.where(np.diff(np.sign(sig_deriv)))[0]
        
        if len(zero_crossings) < 3:
            return np.zeros_like(self.signal)
        
        # Each zero crossing represents π/2 phase
        zc_phase = np.arange(len(zero_crossings)) * np.pi/2
        
        # Estimate local frequency
        local_freq = np.gradient(zc_phase) / np.gradient(zero_crossings)
        
        # Interpolate frequency
        f_freq = interpolate.interp1d(zero_crossings, local_freq, 
                                     kind='linear', fill_value='extrapolate')
        
        # Integrate to get phase
        z_reconstructed = np.zeros(self.n_points)
        for i in range(1, self.n_points):
            if i in zero_crossings:
                idx = np.where(zero_crossings == i)[0][0]
                z_reconstructed[i] = zc_phase[idx]
            else:
                local_f = f_freq(i)
                z_reconstructed[i] = z_reconstructed[i-1] + local_f
        
        return z_reconstructed
    
    def method5_optimization_based(self, window_size=100):
        """Method 5: Local optimization with sliding window"""
        z_reconstructed = np.zeros(self.n_points)
        
        # Initialize with simple peak counting
        z_init = self.method1_peak_counting()
        
        # Sliding window optimization
        for i in range(0, self.n_points - window_size, window_size//2):
            window_end = min(i + window_size, self.n_points)
            window_idx = np.arange(i, window_end)
            
            # Local signal
            local_signal = self.signal[window_idx]
            
            # Objective function
            def objective(z_params):
                # z_params = [amplitude, frequency, phase, offset]
                z_model = z_params[0] * window_idx + z_params[1]
                model_signal = 0.5 + 0.45 * np.sin(z_model/2)**2
                return np.sum((model_signal - local_signal)**2)
            
            # Initial guess from previous fit
            if i == 0:
                x0 = [2*np.pi*10/self.n_points, z_init[i]]
            else:
                x0 = [2*np.pi*10/self.n_points, z_reconstructed[i-1]]
            
            # Optimize
            result = optimize.minimize(objective, x0, method='Nelder-Mead')
            
            # Apply result
            z_local = result.x[0] * window_idx + result.x[1]
            z_reconstructed[window_idx] = z_local
        
        return z_reconstructed
    
    def _smart_unwrap(self, phase):
        """Smart phase unwrapping"""
        unwrapped = np.copy(phase)
        
        for i in range(1, len(phase)):
            diff = phase[i] - phase[i-1]
            
            # Detect phase jumps
            if diff < -np.pi/2:
                unwrapped[i:] += np.pi
            elif diff > np.pi/2:
                unwrapped[i:] -= np.pi
                
        return unwrapped


def evaluate_methods(z_true, z_reconstructed, method_name):
    """Evaluate reconstruction accuracy"""
    # Remove linear trend for comparison
    p_true = np.polyfit(np.arange(len(z_true)), z_true, 1)
    p_recon = np.polyfit(np.arange(len(z_reconstructed)), z_reconstructed, 1)
    
    z_true_detrend = z_true - np.polyval(p_true, np.arange(len(z_true)))
    z_recon_detrend = z_reconstructed - np.polyval(p_recon, np.arange(len(z_reconstructed)))
    
    # Scale to match
    scale = np.std(z_true_detrend) / (np.std(z_recon_detrend) + 1e-10)
    z_recon_scaled = z_recon_detrend * scale
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((z_true_detrend - z_recon_scaled)**2))
    max_error = np.max(np.abs(z_true_detrend - z_recon_scaled))
    correlation = np.corrcoef(z_true_detrend, z_recon_scaled)[0, 1]
    
    return {
        'method': method_name,
        'rmse': rmse,
        'max_error': max_error,
        'correlation': correlation
    }


def main():
    """Run simulation and compare methods"""
    
    # Simulation parameters
    np.random.seed(42)
    
    print("=" * 60)
    print("Mach-Zehnder Interferometer Nonlinear Scan Reconstruction")
    print("=" * 60)
    
    # Create simulator
    sim = MZSimulator(
        n_points=2000,
        n_fringes=15,
        nonlinearity_strength=0.3,
        phase_noise_std=0.02,
        amplitude_noise_std=0.01
    )
    
    # Test different nonlinearity types
    nonlin_types = ['cubic', 'hysteresis', 'exponential']
    
    all_results = []
    
    for nonlin_type in nonlin_types:
        print(f"\n{'='*50}")
        print(f"Testing with {nonlin_type} nonlinearity")
        print(f"{'='*50}")
        
        # Generate data
        t, z_true = sim.generate_nonlinear_scan(nonlin_type)
        mz_signal = sim.generate_mz_signal(z_true)
        
        # Create reconstructor
        reconstructor = ZReconstructor(mz_signal, t)
        
        # Test all methods
        methods = {
            'Peak Counting': reconstructor.method1_peak_counting,
            'Hilbert Transform': reconstructor.method2_hilbert_transform,
            'Arccos Method': reconstructor.method3_arccosine_method,
            'Fringe Tracking': reconstructor.method4_fringe_tracking,
            'Optimization': reconstructor.method5_optimization_based
        }
        
        results = []
        execution_times = {}
        
        for method_name, method_func in methods.items():
            try:
                start_time = time.time()
                z_recon = method_func()
                exec_time = time.time() - start_time
                
                if len(z_recon) > 0 and not np.all(z_recon == 0):
                    result = evaluate_methods(z_true, z_recon, method_name)
                    result['execution_time'] = exec_time
                    result['nonlinearity'] = nonlin_type
                    results.append(result)
                    execution_times[method_name] = exec_time
                    
                    print(f"\n{method_name}:")
                    print(f"  RMSE: {result['rmse']:.6f}")
                    print(f"  Max Error: {result['max_error']:.6f}")
                    print(f"  Correlation: {result['correlation']:.6f}")
                    print(f"  Execution Time: {exec_time*1000:.2f} ms")
                else:
                    print(f"\n{method_name}: Failed")
                    
            except Exception as e:
                print(f"\n{method_name}: Error - {str(e)}")
        
        all_results.extend(results)
        
        # Plot results for this nonlinearity type
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(f'MZ Reconstruction Comparison - {nonlin_type} nonlinearity', fontsize=14)
        
        # Plot 1: Original signal
        axes[0, 0].plot(t, mz_signal, 'b-', linewidth=0.5)
        axes[0, 0].set_xlabel('Time (normalized)')
        axes[0, 0].set_ylabel('MZ Signal')
        axes[0, 0].set_title('Mach-Zehnder Output')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: True vs linear z
        z_linear = t * 2 * np.pi * sim.n_fringes
        axes[0, 1].plot(t, z_true, 'g-', label='True (nonlinear)', linewidth=2)
        axes[0, 1].plot(t, z_linear, 'k--', label='Linear reference', linewidth=1)
        axes[0, 1].set_xlabel('Time (normalized)')
        axes[0, 1].set_ylabel('Phase z (rad)')
        axes[0, 1].set_title('True Phase Evolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3-6: Method comparisons
        plot_positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
        
        for idx, (method_name, method_func) in enumerate(list(methods.items())[:4]):
            row, col = plot_positions[idx]
            
            try:
                z_recon = method_func()
                if len(z_recon) > 0 and not np.all(z_recon == 0):
                    # Scale for visualization
                    scale_factor = np.mean(z_true) / (np.mean(z_recon) + 1e-10)
                    z_recon_scaled = z_recon * scale_factor
                    
                    axes[row, col].plot(t, z_true, 'g-', label='True', linewidth=1, alpha=0.7)
                    axes[row, col].plot(t, z_recon_scaled, 'r--', label='Reconstructed', linewidth=1)
                    axes[row, col].set_xlabel('Time (normalized)')
                    axes[row, col].set_ylabel('Phase z (rad)')
                    axes[row, col].set_title(f'{method_name}')
                    axes[row, col].legend(fontsize=8)
                    axes[row, col].grid(True, alpha=0.3)
            except:
                axes[row, col].text(0.5, 0.5, f'{method_name}\nFailed', 
                                   ha='center', va='center', transform=axes[row, col].transAxes)
        
        plt.tight_layout()
        plt.savefig(f'mz_reconstruction_{nonlin_type}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Summary comparison
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    
    # Find best method for each metric
    if all_results:
        # Convert to numpy arrays for analysis
        methods_list = list(set([r['method'] for r in all_results]))
        nonlin_list = list(set([r['nonlinearity'] for r in all_results]))
        
        print("\nAverage Performance Across All Nonlinearity Types:")
        print("-" * 50)
        
        for method in methods_list:
            method_results = [r for r in all_results if r['method'] == method]
            if method_results:
                avg_rmse = np.mean([r['rmse'] for r in method_results])
                avg_corr = np.mean([r['correlation'] for r in method_results])
                avg_time = np.mean([r['execution_time'] for r in method_results])
                
                print(f"\n{method}:")
                print(f"  Avg RMSE: {avg_rmse:.6f}")
                print(f"  Avg Correlation: {avg_corr:.6f}")
                print(f"  Avg Time: {avg_time*1000:.2f} ms")
        
        # Find best overall method
        best_methods = {}
        for metric in ['rmse', 'correlation']:
            best_val = None
            best_method = None
            
            for method in methods_list:
                method_results = [r for r in all_results if r['method'] == method]
                if method_results:
                    avg_val = np.mean([r[metric] for r in method_results])
                    
                    if metric == 'rmse':
                        if best_val is None or avg_val < best_val:
                            best_val = avg_val
                            best_method = method
                    else:  # correlation
                        if best_val is None or avg_val > best_val:
                            best_val = avg_val
                            best_method = method
            
            best_methods[metric] = (best_method, best_val)
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS:")
        print("="*60)
        print(f"\n✓ Best Accuracy (lowest RMSE): {best_methods['rmse'][0]}")
        print(f"  RMSE = {best_methods['rmse'][1]:.6f}")
        
        print(f"\n✓ Best Correlation: {best_methods['correlation'][0]}")
        print(f"  Correlation = {best_methods['correlation'][1]:.6f}")
        
        # Find best compromise (good accuracy and fast)
        for method in methods_list:
            method_results = [r for r in all_results if r['method'] == method]
            if method_results:
                avg_rmse = np.mean([r['rmse'] for r in method_results])
                avg_time = np.mean([r['execution_time'] for r in method_results])
                
                if avg_rmse < best_methods['rmse'][1] * 1.5 and avg_time < 0.01:  # Within 50% of best and fast
                    print(f"\n✓ Best Compromise (accurate & fast): {method}")
                    print(f"  RMSE = {avg_rmse:.6f}, Time = {avg_time*1000:.2f} ms")
                    break
        
        print("\n" + "="*60)
        print("Method Selection Guide:")
        print("-" * 60)
        print("• For HIGHEST ACCURACY: Use Fringe Tracking or Optimization methods")
        print("• For REAL-TIME processing: Use Peak Counting or Hilbert Transform")
        print("• For NOISY signals: Use Fringe Tracking with smoothing")
        print("• For SIMPLE implementation: Use Peak Counting with cubic interpolation")
        print("=" * 60)


if __name__ == "__main__":
    main()
