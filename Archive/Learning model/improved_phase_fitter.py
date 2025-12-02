"""
Improved Nonlinear Phase Fitter with Better Diagnostics
========================================================
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit, differential_evolution
from scipy.signal import hilbert
import warnings
warnings.filterwarnings('ignore')


def simple_fit_mi_signal(time, mi_signal, wavelength=632.8e-9, verbose=True):
    """
    Simple direct fitting approach for MI signals.
    
    This uses a physics-based model:
    I(t) = I0 * [1 + V * cos(phi(t))]
    
    where phi(t) is estimated from the signal oscillations.
    """
    
    if verbose:
        print("\n" + "=" * 70)
        print("SIMPLE PHYSICS-BASED FITTING")
        print("=" * 70)
    
    time = np.array(time)
    mi_signal = np.array(mi_signal)
    
    if verbose:
        print(f"\nSignal info:")
        print(f"  Points: {len(time)}")
        print(f"  Time range: {time[0]:.6f} to {time[-1]:.6f} s")
        print(f"  MI range: {mi_signal.min():.6f} to {mi_signal.max():.6f}")
    
    # Step 1: Extract phase using Hilbert transform
    if verbose:
        print(f"\nStep 1: Extracting phase...")
    
    # Center the signal
    mi_centered = mi_signal - np.mean(mi_signal)
    
    # Hilbert transform to get analytic signal
    analytic_signal = hilbert(mi_centered)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    
    # Remove linear trend (represents average velocity)
    phase_coeffs = np.polyfit(time, instantaneous_phase, 1)
    phase_detrended = instantaneous_phase - np.polyval(phase_coeffs, time)
    
    if verbose:
        print(f"  Phase range: {instantaneous_phase.min():.2f} to {instantaneous_phase.max():.2f} rad")
        print(f"  Phase variation: {np.ptp(instantaneous_phase):.2f} rad ({np.ptp(instantaneous_phase)/(2*np.pi):.2f} cycles)")
    
    # Step 2: Fit the signal model
    if verbose:
        print(f"\nStep 2: Fitting signal model...")
    
    def signal_model(t, I0, V, phase_scale, phase_offset):
        """MI signal model: I = I0[1 + V*cos(phi)]"""
        # Use extracted phase but allow scaling
        idx = np.searchsorted(time, t)
        idx = np.clip(idx, 0, len(instantaneous_phase) - 1)
        phi = phase_scale * instantaneous_phase[idx] + phase_offset
        return I0 * (1 + V * np.cos(phi))
    
    # Initial guess
    I0_guess = np.mean(mi_signal)
    V_guess = (np.max(mi_signal) - np.min(mi_signal)) / (2 * I0_guess)
    V_guess = np.clip(V_guess, 0.1, 1.0)
    
    p0 = [I0_guess, V_guess, 1.0, 0.0]
    
    if verbose:
        print(f"  Initial guess: I0={I0_guess:.4f}, V={V_guess:.4f}")
    
    try:
        # Fit with bounds
        bounds = (
            [0, 0, 0.5, -2*np.pi],  # lower bounds
            [2*np.max(mi_signal), 1.0, 2.0, 2*np.pi]  # upper bounds
        )
        
        popt, pcov = curve_fit(signal_model, time, mi_signal, p0=p0, 
                              bounds=bounds, maxfev=10000)
        
        I0_fit, V_fit, phase_scale_fit, phase_offset_fit = popt
        
        # Generate fitted signal
        mi_fitted = signal_model(time, *popt)
        
        # Calculate residuals and metrics
        residuals = mi_signal - mi_fitted
        rmse = np.sqrt(np.mean(residuals**2))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((mi_signal - np.mean(mi_signal))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        if verbose:
            print(f"\n  Fit results:")
            print(f"    I0 (mean intensity): {I0_fit:.6f}")
            print(f"    V (visibility): {V_fit:.6f}")
            print(f"    Phase scale: {phase_scale_fit:.6f}")
            print(f"    Phase offset: {phase_offset_fit:.6f} rad")
            print(f"    RMSE: {rmse:.6f}")
            print(f"    R²: {r_squared:.6f}")
        
        result = {
            'fitted_signal': mi_fitted,
            'residuals': residuals,
            'phase': phase_scale_fit * instantaneous_phase + phase_offset_fit,
            'rmse': rmse,
            'r_squared': r_squared,
            'I0': I0_fit,
            'visibility': V_fit,
            'phase_scale': phase_scale_fit,
            'phase_offset': phase_offset_fit,
            'success': True
        }
        
        if r_squared < 0.5:
            if verbose:
                print(f"\n  ⚠ Warning: Low R² ({r_squared:.4f}). Fit may be poor.")
        elif r_squared > 0.95:
            if verbose:
                print(f"\n  ✓ Excellent fit! R² = {r_squared:.6f}")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"\n  ✗ Fitting failed: {e}")
        
        # Return simple average as fallback
        return {
            'fitted_signal': np.full_like(mi_signal, np.mean(mi_signal)),
            'residuals': mi_signal - np.mean(mi_signal),
            'phase': instantaneous_phase,
            'rmse': np.std(mi_signal),
            'r_squared': 0.0,
            'I0': np.mean(mi_signal),
            'visibility': 0.0,
            'phase_scale': 1.0,
            'phase_offset': 0.0,
            'success': False
        }


def plot_simple_fit_result(time, mi_signal, fit_result, 
                           save_path='./DATA/improved_fit_result.png'):
    """Plot fitting results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Signal fit
    ax = axes[0, 0]
    ax.plot(time, mi_signal, 'b-', linewidth=2, alpha=0.7, label='Original')
    ax.plot(time, fit_result['fitted_signal'], 'r--', linewidth=2, alpha=0.7, label='Fitted')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('MI Signal', fontsize=11)
    ax.set_title(f'Signal Fit (R² = {fit_result["r_squared"]:.6f})', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. Residuals
    ax = axes[0, 1]
    ax.plot(time, fit_result['residuals'], 'g-', linewidth=1, alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.fill_between(time, fit_result['residuals'], alpha=0.3)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Residuals', fontsize=11)
    ax.set_title(f'Residuals (RMSE = {fit_result["rmse"]:.6f})', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. Actual vs Fitted
    ax = axes[1, 0]
    ax.scatter(mi_signal, fit_result['fitted_signal'], alpha=0.5, s=20)
    min_val = min(mi_signal.min(), fit_result['fitted_signal'].min())
    max_val = max(mi_signal.max(), fit_result['fitted_signal'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect fit')
    ax.set_xlabel('Actual MI Signal', fontsize=11)
    ax.set_ylabel('Fitted MI Signal', fontsize=11)
    ax.set_title('Actual vs Fitted', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # 4. Residual histogram
    ax = axes[1, 1]
    ax.hist(fit_result['residuals'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual Value', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    # Ensure the directory for the save path exists, create it if it doesn't
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {save_path}")
    
    return fig


# Example usage
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("IMPROVED FITTER TEST")
    print("=" * 70)
    
    # Generate test signal
    np.random.seed(42)
    wavelength = 632.8e-9
    
    time = np.linspace(0, 0.2, 200)
    
    # Piezo with nonlinearity
    piezo = 1.5e-6 * (time / 0.2)  # Linear ramp
    piezo += 0.2 * 1.5e-6 * np.sin(10 * np.pi * time)  # Add nonlinearity
    
    # MI signal
    phase = 4 * np.pi * piezo / wavelength
    mi_signal = 1 + 0.9 * np.cos(phase)
    mi_signal += 0.02 * np.random.randn(len(time))
    
    print(f"\nTest signal generated:")
    print(f"  {len(time)} points")
    print(f"  MI range: {mi_signal.min():.4f} to {mi_signal.max():.4f}")
    
    # Fit the signal
    fit_result = simple_fit_mi_signal(time, mi_signal, wavelength=wavelength)
    
    # Plot results
    plot_simple_fit_result(time, mi_signal, fit_result)
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


"""
Edge-Effect-Free MI Signal Fitter
==================================
Eliminates edge effects by using signal padding and windowing.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import hilbert, windows
import warnings
warnings.filterwarnings('ignore')


def fit_mi_signal_no_edges(time, mi_signal, wavelength=1389e-9, 
                            edge_trim=0.02, use_padding=True, verbose=True):
    """
    Fit MI signal with edge effect mitigation.
    
    Parameters:
    -----------
    time : array
        Time values
    mi_signal : array
        MI signal values
    wavelength : float
        Laser wavelength in meters
    edge_trim : float
        Additional fraction to trim from edges (0 to 0.2)
        Example: 0.05 = trim 5% from each edge
    use_padding : bool
        Use symmetric padding to reduce edge effects
    verbose : bool
        Print detailed output
    
    Returns:
    --------
    result : dict
        Fitting results with edge effects minimized
    """
    
    if verbose:
        print("\n" + "=" * 70)
        print("EDGE-EFFECT-FREE FITTING")
        print("=" * 70)
    
    time = np.array(time)
    mi_signal = np.array(mi_signal)
    n_original = len(time)
    
    if verbose:
        print(f"\nOriginal signal:")
        print(f"  Points: {len(time)}")
        print(f"  Duration: {time[-1] - time[0]:.6f} s")
        print(f"  MI range: {mi_signal.min():.6f} to {mi_signal.max():.6f}")
    
    # Step 1: Apply edge trimming if requested
    if edge_trim > 0:
        n_trim = int(len(time) * edge_trim)
        if n_trim > 0:
            time = time[n_trim:-n_trim]
            mi_signal = mi_signal[n_trim:-n_trim]
            if verbose:
                print(f"\nEdge trimming:")
                print(f"  Trimmed {edge_trim*100:.1f}% from each edge")
                print(f"  New points: {len(time)}")
                print(f"  New duration: {time[-1] - time[0]:.6f} s")
    
    # Step 2: Pad signal to reduce boundary effects
    if use_padding:
        pad_length = len(mi_signal) // 4  # Pad 25% on each side
        
        # Symmetric padding (mirror signal at boundaries)
        mi_padded = np.pad(mi_signal, pad_length, mode='reflect')
        
        # Extend time array
        dt = np.mean(np.diff(time))
        time_padded = np.concatenate([
            time[0] - dt * np.arange(pad_length, 0, -1),
            time,
            time[-1] + dt * np.arange(1, pad_length + 1)
        ])
        
        if verbose:
            print(f"\nPadding applied:")
            print(f"  Pad length: {pad_length} points ({pad_length*dt:.6f} s)")
            print(f"  Padded length: {len(mi_padded)} points")
    else:
        mi_padded = mi_signal.copy()
        time_padded = time.copy()
        pad_length = 0
    
    # Step 3: Extract phase with Hilbert transform
    if verbose:
        print(f"\nPhase extraction:")
    
    # Center the signal
    mi_centered = mi_padded - np.mean(mi_padded)
    
    # Apply window to further reduce edge effects
    window = windows.tukey(len(mi_centered), alpha=0.1)
    mi_windowed = mi_centered * window
    
    # Hilbert transform
    analytic_signal = hilbert(mi_windowed)
    phase_padded = np.unwrap(np.angle(analytic_signal))
    
    # Extract the central portion (remove padding)
    if use_padding:
        phase = phase_padded[pad_length:-pad_length]
    else:
        phase = phase_padded
    
    if verbose:
        print(f"  Phase range: {phase.min():.2f} to {phase.max():.2f} rad")
        print(f"  Phase span: {np.ptp(phase):.2f} rad ({np.ptp(phase)/(2*np.pi):.2f} cycles)")
    
    # Step 4: Fit the signal model
    if verbose:
        print(f"\nFitting signal model...")
    
    def signal_model(t, I0, V, phase_scale, phase_offset):
        """MI signal model: I = I0[1 + V*cos(phi)]"""
        idx = np.searchsorted(time, t)
        idx = np.clip(idx, 0, len(phase) - 1)
        phi = phase_scale * phase[idx] + phase_offset
        return I0 * (1 + V * np.cos(phi))
    
    # Initial guess
    I0_guess = np.mean(mi_signal)
    V_guess = (np.max(mi_signal) - np.min(mi_signal)) / (2 * I0_guess)
    V_guess = np.clip(V_guess, 0.1, 1.0)
    
    p0 = [I0_guess, V_guess, 1.0, 0.0]
    
    try:
        bounds = (
            [0, 0, 0.5, -2*np.pi],
            [2*np.max(mi_signal), 1.0, 2.0, 2*np.pi]
        )
        
        popt, pcov = curve_fit(signal_model, time, mi_signal, p0=p0, 
                              bounds=bounds, maxfev=10000)
        
        I0_fit, V_fit, phase_scale_fit, phase_offset_fit = popt
        
        # Generate fitted signal
        mi_fitted = signal_model(time, *popt)
        
        # Calculate metrics
        residuals = mi_signal - mi_fitted
        rmse = np.sqrt(np.mean(residuals**2))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((mi_signal - np.mean(mi_signal))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Check edge residuals specifically
        edge_points = max(10, len(residuals) // 20)
        left_edge_rmse = np.sqrt(np.mean(residuals[:edge_points]**2))
        right_edge_rmse = np.sqrt(np.mean(residuals[-edge_points:]**2))
        center_rmse = np.sqrt(np.mean(residuals[edge_points:-edge_points]**2))
        
        if verbose:
            print(f"\n  Fit results:")
            print(f"    I0: {I0_fit:.6f}")
            print(f"    Visibility: {V_fit:.6f}")
            print(f"    Phase scale: {phase_scale_fit:.6f}")
            print(f"    RMSE (overall): {rmse:.6f}")
            print(f"    RMSE (left edge): {left_edge_rmse:.6f}")
            print(f"    RMSE (center): {center_rmse:.6f}")
            print(f"    RMSE (right edge): {right_edge_rmse:.6f}")
            print(f"    R²: {r_squared:.6f}")
            
            edge_ratio = max(left_edge_rmse, right_edge_rmse) / center_rmse
            if edge_ratio > 2.0:
                print(f"\n  ⚠ Edge residuals {edge_ratio:.1f}x larger than center")
                print(f"  → Try increasing edge_trim or check signal quality")
            else:
                print(f"\n  ✓ Edge effects under control (ratio: {edge_ratio:.2f})")
        
        result = {
            'time': time,
            'fitted_signal': mi_fitted,
            'residuals': residuals,
            'phase': phase_scale_fit * phase + phase_offset_fit,
            'rmse': rmse,
            'left_edge_rmse': left_edge_rmse,
            'center_rmse': center_rmse,
            'right_edge_rmse': right_edge_rmse,
            'r_squared': r_squared,
            'I0': I0_fit,
            'visibility': V_fit,
            'phase_scale': phase_scale_fit,
            'phase_offset': phase_offset_fit,
            'success': True
        }
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"\n  ✗ Fitting failed: {e}")
        
        return {
            'time': time,
            'fitted_signal': np.full_like(mi_signal, np.mean(mi_signal)),
            'residuals': mi_signal - np.mean(mi_signal),
            'phase': phase,
            'rmse': np.std(mi_signal),
            'left_edge_rmse': np.nan,
            'center_rmse': np.nan,
            'right_edge_rmse': np.nan,
            'r_squared': 0.0,
            'I0': np.mean(mi_signal),
            'visibility': 0.0,
            'success': False
        }


def plot_edge_analysis(time, mi_signal, fit_result, 
                       save_path='./DATA/edge_analysis.png'):
    """Create detailed edge effect analysis plot."""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Full signal fit
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time, mi_signal, 'b-', linewidth=1.5, alpha=0.7, label='Original')
    ax1.plot(time, fit_result['fitted_signal'], 'r--', linewidth=1.5, alpha=0.7, label='Fitted')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('MI Signal', fontsize=11)
    ax1.set_title(f'Signal Fit (R² = {fit_result["r_squared"]:.6f})', 
                 fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Highlight edge regions
    edge_points = max(10, len(time) // 20)
    ax1.axvspan(time[0], time[edge_points], alpha=0.2, color='orange', label='Edge regions')
    ax1.axvspan(time[-edge_points], time[-1], alpha=0.2, color='orange')
    
    # 2. Left edge zoom
    ax2 = fig.add_subplot(gs[1, 0])
    n_zoom = min(50, len(time) // 4)
    ax2.plot(time[:n_zoom], mi_signal[:n_zoom], 'b-', linewidth=2, alpha=0.7, label='Original')
    ax2.plot(time[:n_zoom], fit_result['fitted_signal'][:n_zoom], 'r--', 
            linewidth=2, alpha=0.7, label='Fitted')
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.set_ylabel('MI Signal', fontsize=10)
    ax2.set_title(f'Left Edge (RMSE={fit_result["left_edge_rmse"]:.6f})', 
                 fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Center zoom
    ax3 = fig.add_subplot(gs[1, 1])
    center_idx = len(time) // 2
    idx_range = n_zoom // 2
    idx_start = center_idx - idx_range
    idx_end = center_idx + idx_range
    ax3.plot(time[idx_start:idx_end], mi_signal[idx_start:idx_end], 'b-', 
            linewidth=2, alpha=0.7, label='Original')
    ax3.plot(time[idx_start:idx_end], fit_result['fitted_signal'][idx_start:idx_end], 
            'r--', linewidth=2, alpha=0.7, label='Fitted')
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('MI Signal', fontsize=10)
    ax3.set_title(f'Center (RMSE={fit_result["center_rmse"]:.6f})', 
                 fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Right edge zoom
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(time[-n_zoom:], mi_signal[-n_zoom:], 'b-', linewidth=2, alpha=0.7, label='Original')
    ax4.plot(time[-n_zoom:], fit_result['fitted_signal'][-n_zoom:], 'r--', 
            linewidth=2, alpha=0.7, label='Fitted')
    ax4.set_xlabel('Time (s)', fontsize=10)
    ax4.set_ylabel('MI Signal', fontsize=10)
    ax4.set_title(f'Right Edge (RMSE={fit_result["right_edge_rmse"]:.6f})', 
                 fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. Residuals over time
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(time, fit_result['residuals'], 'g-', linewidth=1, alpha=0.7)
    ax5.axhline(y=0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
    ax5.fill_between(time, fit_result['residuals'], alpha=0.3, color='green')
    ax5.set_xlabel('Time (s)', fontsize=10)
    ax5.set_ylabel('Residuals', fontsize=10)
    ax5.set_title(f'Residuals (RMSE = {fit_result["rmse"]:.6f})', 
                 fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Highlight edge regions
    ax5.axvspan(time[0], time[edge_points], alpha=0.2, color='orange')
    ax5.axvspan(time[-edge_points], time[-1], alpha=0.2, color='orange')
    
    # 6. Residual spatial analysis
    ax6 = fig.add_subplot(gs[2, 1])
    n_bins = 20
    bin_edges = np.linspace(0, len(time), n_bins + 1)
    bin_rmse = []
    bin_centers = []
    
    for i in range(n_bins):
        start_idx = int(bin_edges[i])
        end_idx = int(bin_edges[i + 1])
        if end_idx > start_idx:
            bin_residuals = fit_result['residuals'][start_idx:end_idx]
            bin_rmse.append(np.sqrt(np.mean(bin_residuals**2)))
            bin_centers.append((time[start_idx] + time[min(end_idx-1, len(time)-1)]) / 2)
    
    ax6.bar(bin_centers, bin_rmse, width=np.diff(bin_centers)[0] * 0.8, 
           alpha=0.7, color='purple', edgecolor='black')
    ax6.set_xlabel('Time (s)', fontsize=10)
    ax6.set_ylabel('RMSE', fontsize=10)
    ax6.set_title('Residual RMSE by Position', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Residual histogram
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.hist(fit_result['residuals'], bins=50, edgecolor='black', alpha=0.7, color='cyan')
    ax7.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax7.set_xlabel('Residual Value', fontsize=10)
    ax7.set_ylabel('Frequency', fontsize=10)
    ax7.set_title('Residual Distribution', fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # Ensure the directory for the save path exists, create it if it doesn't
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nEdge analysis plot saved: {save_path}")
    
    return fig

"""
Aggressive Edge Effect Removal
===============================
For stubborn edge effects that padding alone can't fix.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import hilbert, windows, savgol_filter
import warnings
warnings.filterwarnings('ignore')


def fit_with_aggressive_edge_removal(time, mi_signal, wavelength=632.8e-9, 
                                     edge_discard_fraction=0.05, verbose=True):
    """
    Fit MI signal with aggressive edge removal.
    
    Simply discards the problematic edges entirely and fits only the clean center.
    
    Parameters:
    -----------
    time : array
        Time values
    mi_signal : array
        MI signal values
    wavelength : float
        Laser wavelength in meters
    edge_discard_fraction : float
        Fraction to discard from EACH edge (default 0.05 = 5% from each side)
        Total discarded = 2 * edge_discard_fraction
    verbose : bool
        Print output
    
    Returns:
    --------
    result : dict
        Fitting results on center portion only
    """
    
    if verbose:
        print("\n" + "=" * 70)
        print("AGGRESSIVE EDGE REMOVAL FITTING")
        print("=" * 70)
    
    time = np.array(time)
    mi_signal = np.array(mi_signal)
    
    if verbose:
        print(f"\nOriginal signal:")
        print(f"  Points: {len(time)}")
        print(f"  Duration: {time[-1] - time[0]:.6f} s")
    
    # Aggressively discard edges
    n_total = len(time)
    n_discard = int(n_total * edge_discard_fraction)
    
    if n_discard < 1:
        n_discard = 1
    
    time_center = time[n_discard:-n_discard]
    mi_center = mi_signal[n_discard:-n_discard]
    
    if verbose:
        print(f"\nEdge removal:")
        print(f"  Discarding: {edge_discard_fraction*100:.1f}% from each edge")
        print(f"  Discarded points: {n_discard} from each side")
        print(f"  Remaining points: {len(time_center)}")
        print(f"  Remaining duration: {time_center[-1] - time_center[0]:.6f} s")
        print(f"  Time lost: {time[0]:.6f} to {time_center[0]:.6f} s (left)")
        print(f"              {time_center[-1]:.6f} to {time[-1]:.6f} s (right)")
    
    # Now fit the center portion with padding to avoid creating NEW edge effects
    pad_length = len(mi_center) // 10
    mi_padded = np.pad(mi_center, pad_length, mode='reflect')
    
    dt = np.mean(np.diff(time_center))
    time_padded = np.concatenate([
        time_center[0] - dt * np.arange(pad_length, 0, -1),
        time_center,
        time_center[-1] + dt * np.arange(1, pad_length + 1)
    ])
    
    # Extract phase with Hilbert
    mi_centered = mi_padded - np.mean(mi_padded)
    window = windows.tukey(len(mi_centered), alpha=0.1)
    mi_windowed = mi_centered * window
    
    analytic_signal = hilbert(mi_windowed)
    phase_padded = np.unwrap(np.angle(analytic_signal))
    phase = phase_padded[pad_length:-pad_length]
    
    if verbose:
        print(f"\nPhase extraction (on center only):")
        print(f"  Phase range: {phase.min():.2f} to {phase.max():.2f} rad")
        print(f"  Phase span: {np.ptp(phase):.2f} rad ({np.ptp(phase)/(2*np.pi):.2f} cycles)")
    
    # Fit the model
    def signal_model(t, I0, V, phase_scale, phase_offset):
        idx = np.searchsorted(time_center, t)
        idx = np.clip(idx, 0, len(phase) - 1)
        phi = phase_scale * phase[idx] + phase_offset
        return I0 * (1 + V * np.cos(phi))
    
    I0_guess = np.mean(mi_center)
    V_guess = (np.max(mi_center) - np.min(mi_center)) / (2 * I0_guess)
    V_guess = np.clip(V_guess, 0.1, 1.0)
    
    p0 = [I0_guess, V_guess, 1.0, 0.0]
    
    try:
        bounds = (
            [0, 0, 0.5, -2*np.pi],
            [2*np.max(mi_center), 1.0, 2.0, 2*np.pi]
        )
        
        popt, pcov = curve_fit(signal_model, time_center, mi_center, p0=p0, 
                              bounds=bounds, maxfev=10000)
        
        I0_fit, V_fit, phase_scale_fit, phase_offset_fit = popt
        
        mi_fitted = signal_model(time_center, *popt)
        
        residuals = mi_center - mi_fitted
        rmse = np.sqrt(np.mean(residuals**2))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((mi_center - np.mean(mi_center))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        if verbose:
            print(f"\nFit results:")
            print(f"  I0: {I0_fit:.6f}")
            print(f"  Visibility: {V_fit:.6f}")
            print(f"  Phase scale: {phase_scale_fit:.6f}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  R²: {r_squared:.6f}")
        
        # Check residual uniformity
        n_check = len(residuals) // 20
        if n_check > 0:
            left_rmse = np.sqrt(np.mean(residuals[:n_check]**2))
            center_rmse = np.sqrt(np.mean(residuals[n_check:-n_check]**2))
            right_rmse = np.sqrt(np.mean(residuals[-n_check:]**2))
            
            if verbose:
                print(f"\n  Residual uniformity (on remaining signal):")
                print(f"    Left section RMSE: {left_rmse:.6f}")
                print(f"    Center section RMSE: {center_rmse:.6f}")
                print(f"    Right section RMSE: {right_rmse:.6f}")
                
                max_rmse = max(left_rmse, right_rmse)
                if center_rmse > 0:
                    ratio = max_rmse / center_rmse
                    print(f"    Edge/Center ratio: {ratio:.2f}")
                    
                    if ratio < 1.5:
                        print(f"    ✓ Uniform! Edge effects eliminated.")
                    else:
                        print(f"    ⚠ Still some variation (try larger edge_discard_fraction)")
        
        return {
            'time': time_center,
            'time_original': time,
            'fitted_signal': mi_fitted,
            'original_signal': mi_signal,
            'center_signal': mi_center,
            'residuals': residuals,
            'phase': phase_scale_fit * phase + phase_offset_fit,
            'rmse': rmse,
            'r_squared': r_squared,
            'I0': I0_fit,
            'visibility': V_fit,
            'edge_discard_fraction': edge_discard_fraction,
            'discarded_points_per_side': n_discard,
            'success': True
        }
        
    except Exception as e:
        if verbose:
            print(f"\n  ✗ Fitting failed: {e}")
        
        return {
            'time': time_center,
            'fitted_signal': np.full_like(mi_center, np.mean(mi_center)),
            'residuals': mi_center - np.mean(mi_center),
            'rmse': np.std(mi_center),
            'r_squared': 0.0,
            'success': False
        }


def plot_before_after_edge_removal(time_orig, mi_orig, fit_result,
                                   save_path='./DATA/edge_removal_comparison.png'):
    """
    Plot comparison showing original signal with edges vs fitted center.
    """
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 1. Original full signal
    ax = axes[0, 0]
    ax.plot(time_orig, mi_orig, 'b-', linewidth=1.5, alpha=0.7)
    
    # Highlight discarded regions
    n_discard = fit_result['discarded_points_per_side']
    ax.axvspan(time_orig[0], time_orig[n_discard], alpha=0.3, color='red', label='Discarded')
    ax.axvspan(time_orig[-n_discard], time_orig[-1], alpha=0.3, color='red')
    ax.axvspan(time_orig[n_discard], time_orig[-n_discard], alpha=0.1, color='green', label='Used for fit')
    
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('MI Signal', fontsize=11)
    ax.set_title('Original Signal with Edge Removal Zones', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. Center portion fitted
    ax = axes[0, 1]
    ax.plot(fit_result['time'], fit_result['center_signal'], 'b-', 
           linewidth=1.5, alpha=0.7, label='Original (center)')
    ax.plot(fit_result['time'], fit_result['fitted_signal'], 'r--', 
           linewidth=1.5, alpha=0.7, label='Fitted')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('MI Signal', fontsize=11)
    ax.set_title(f'Fitted Center (R² = {fit_result["r_squared"]:.6f})', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. Left edge zoom (original - showing the problem)
    ax = axes[1, 0]
    n_zoom = min(100, len(time_orig) // 10)
    ax.plot(time_orig[:n_zoom], mi_orig[:n_zoom], 'b-', linewidth=2, alpha=0.7)
    ax.axvline(x=time_orig[n_discard], color='r', linestyle='--', linewidth=2, label='Discard line')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('MI Signal', fontsize=10)
    ax.set_title('Left Edge (ORIGINAL - Problematic)', fontsize=11, fontweight='bold', color='red')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 4. Right edge zoom (original - showing the problem)
    ax = axes[1, 1]
    ax.plot(time_orig[-n_zoom:], mi_orig[-n_zoom:], 'b-', linewidth=2, alpha=0.7)
    ax.axvline(x=time_orig[-n_discard], color='r', linestyle='--', linewidth=2, label='Discard line')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('MI Signal', fontsize=10)
    ax.set_title('Right Edge (ORIGINAL - Problematic)', fontsize=11, fontweight='bold', color='red')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 5. Residuals on center portion
    ax = axes[2, 0]
    ax.plot(fit_result['time'], fit_result['residuals'], 'g-', linewidth=1, alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1.5)
    ax.fill_between(fit_result['time'], fit_result['residuals'], alpha=0.3, color='green')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Residuals', fontsize=10)
    ax.set_title(f'Residuals (Center Only, RMSE = {fit_result["rmse"]:.6f})', 
                fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 6. Residual histogram
    ax = axes[2, 1]
    ax.hist(fit_result['residuals'], bins=50, edgecolor='black', alpha=0.7, color='cyan')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual Value', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Residual Distribution (Center Only)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Ensure the directory for the save path exists, create it if it doesn't
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved: {save_path}")
    
    return fig

def fit_with_adaptive_outlier_removal(time, mi_signal, snom_signal, wavelength=632.8e-9,
                                     outlier_threshold=3.0, max_iterations=5,
                                     max_removal_fraction=0.15, verbose=True):
    """
    Fit MI signal with adaptive outlier removal.
    
    Iteratively:
    1. Fit the signal
    2. Find points with large residuals (outliers)
    3. Remove those points
    4. Refit
    5. Repeat until convergence or max iterations
    
    Parameters:
    -----------
    time : array
        Time values
    mi_signal : array
        MI signal values
    wavelength : float
        Laser wavelength in meters
    outlier_threshold : float
        Number of standard deviations for outlier detection
        Larger = more tolerant (removes less)
        Typical: 2.5-3.5
    max_iterations : int
        Maximum number of removal iterations
    max_removal_fraction : float
        Maximum fraction of points to remove (safety limit)
    verbose : bool
        Print detailed output
    
    Returns:
    --------
    result : dict
        Fitting results with outliers removed
    """
    
    if verbose:
        print("\n" + "=" * 70)
        print("ADAPTIVE OUTLIER REMOVAL FITTING")
        print("=" * 70)
        print(f"\nParameters:")
        print(f"  Outlier threshold: {outlier_threshold} × std dev")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Max removal: {max_removal_fraction*100:.1f}% of points")
    
    time = np.array(time)
    mi_signal = np.array(mi_signal)
    snom_signal = np.array(snom_signal)
    
    # Track which points to keep
    keep_mask = np.ones(len(time), dtype=bool)
    n_original = len(time)
    
    if verbose:
        print(f"\nOriginal signal:")
        print(f"  Points: {n_original}")
        print(f"  Duration: {time[-1] - time[0]:.6f} s")
    
    # Iterative fitting and outlier removal
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}")
            print(f"{'='*70}")
        
        # Current data (only kept points)
        time_current = time[keep_mask]
        mi_current = mi_signal[keep_mask]
        snom_current = snom_signal[keep_mask]
        n_current = len(time_current)
        
        if verbose:
            print(f"  Current points: {n_current} ({n_current/n_original*100:.1f}% of original)")
        
        # Fit current data
        fit_result = _fit_single_iteration(time_current, mi_current, wavelength, verbose=False)
        
        if not fit_result['success']:
            if verbose:
                print(f"  ✗ Fitting failed at iteration {iteration + 1}")
            break
        
        residuals = fit_result['residuals']
        rmse = fit_result['rmse']
        
        if verbose:
            print(f"  RMSE: {rmse:.6f}")
            print(f"  R²: {fit_result['r_squared']:.6f}")
        
        # Detect outliers based on residuals
        residual_std = np.std(residuals)
        outlier_threshold_value = outlier_threshold * residual_std
        
        outlier_mask_current = np.abs(residuals) > outlier_threshold_value
        n_outliers = np.sum(outlier_mask_current)
        
        if verbose:
            print(f"  Residual std: {residual_std:.6f}")
            print(f"  Outlier threshold: {outlier_threshold_value:.6f}")
            print(f"  Outliers found: {n_outliers}")
        
        if n_outliers == 0:
            if verbose:
                print(f"\n  ✓ Converged! No more outliers found.")
            break
        
        # Check if we'd remove too many points
        n_would_remove = np.sum(~keep_mask) + n_outliers
        if n_would_remove / n_original > max_removal_fraction:
            if verbose:
                print(f"\n  ⚠ Would exceed max removal limit ({n_would_remove/n_original*100:.1f}% > {max_removal_fraction*100:.1f}%)")
                print(f"  Stopping at iteration {iteration + 1}")
            break
        
        # Update global keep mask
        # Map outliers from current indices back to original indices
        current_to_original = np.where(keep_mask)[0]
        outlier_original_indices = current_to_original[outlier_mask_current]
        keep_mask[outlier_original_indices] = False
        
        if verbose:
            # Identify where outliers are
            outlier_times = time[outlier_original_indices]
            if len(outlier_times) > 0:
                print(f"  Removed points at times:")
                if len(outlier_times) <= 10:
                    for t in outlier_times:
                        print(f"    {t:.6f} s")
                else:
                    print(f"    {outlier_times[0]:.6f} to {outlier_times[-1]:.6f} s ({len(outlier_times)} points)")
        
        # Check if this is the last iteration
        if iteration == max_iterations - 1:
            if verbose:
                print(f"\n  ⚠ Reached max iterations ({max_iterations})")
            break
    
    # Final fit on cleaned data
    time_final = time[keep_mask]
    mi_final = mi_signal[keep_mask]
    snom_final = snom_signal[keep_mask]

    if verbose:
        print(f"\n{'='*70}")
        print("FINAL RESULT")
        print(f"{'='*70}")
        print(f"  Total points removed: {n_original - len(time_final)} ({(n_original - len(time_final))/n_original*100:.1f}%)")
        print(f"  Points remaining: {len(time_final)} ({len(time_final)/n_original*100:.1f}%)")
    
    final_fit = _fit_single_iteration(time_final, mi_final, wavelength, verbose=verbose)
    
    # Add metadata
    final_fit['time'] = time_final
    final_fit['mi_signal'] = mi_final
    final_fit['snom_signal'] = snom_final
    final_fit['time_original'] = time
    final_fit['mi_signal_original'] = mi_signal
    final_fit['snom_signal_original'] = snom_signal
    final_fit['keep_mask'] = keep_mask
    final_fit['removed_mask'] = ~keep_mask
    final_fit['n_removed'] = n_original - len(time_final)
    final_fit['removal_fraction'] = (n_original - len(time_final)) / n_original
    
    # Analyze where removals occurred
    if np.any(~keep_mask):
        removed_indices = np.where(~keep_mask)[0]
        if verbose:
            print(f"\n  Removal distribution:")
            print(f"    First removed index: {removed_indices[0]} (time {time[removed_indices[0]]:.6f} s)")
            print(f"    Last removed index: {removed_indices[-1]} (time {time[removed_indices[-1]]:.6f} s)")
            
            # Count removals in different regions
            n_total = len(time)
            edge_size = n_total // 10
            
            left_edge = removed_indices < edge_size
            right_edge = removed_indices >= (n_total - edge_size)
            center = ~(left_edge | right_edge)
            
            print(f"    Removed from left edge (0-10%): {np.sum(left_edge)}")
            print(f"    Removed from center (10-90%): {np.sum(center)}")
            print(f"    Removed from right edge (90-100%): {np.sum(right_edge)}")
    
    return final_fit


def _fit_single_iteration(time, mi_signal, wavelength, verbose=False):
    """Helper function to fit a single iteration."""
    
    # Pad for Hilbert
    pad_length = len(mi_signal) // 10
    mi_padded = np.pad(mi_signal, pad_length, mode='reflect')
    
    dt = np.mean(np.diff(time))
    time_padded = np.concatenate([
        time[0] - dt * np.arange(pad_length, 0, -1),
        time,
        time[-1] + dt * np.arange(1, pad_length + 1)
    ])
    
    # Phase extraction
    mi_centered = mi_padded - np.mean(mi_padded)
    window = windows.tukey(len(mi_centered), alpha=0.1)
    mi_windowed = mi_centered * window
    
    analytic_signal = hilbert(mi_windowed)
    phase_padded = np.unwrap(np.angle(analytic_signal))
    phase = phase_padded[pad_length:-pad_length]
    
    # Fit model
    def signal_model(t, I0, V, phase_scale, phase_offset):
        idx = np.searchsorted(time, t)
        idx = np.clip(idx, 0, len(phase) - 1)
        phi = phase_scale * phase[idx] + phase_offset
        return I0 * (1 + V * np.cos(phi))
    
    I0_guess = np.mean(mi_signal)
    V_guess = (np.max(mi_signal) - np.min(mi_signal)) / (2 * I0_guess)
    V_guess = np.clip(V_guess, 0.1, 1.0)
    
    p0 = [I0_guess, V_guess, 1.0, 0.0]
    
    try:
        bounds = (
            [0, 0, 0.5, -2*np.pi],
            [2*np.max(mi_signal), 1.0, 2.0, 2*np.pi]
        )
        
        popt, pcov = curve_fit(signal_model, time, mi_signal, p0=p0, 
                              bounds=bounds, maxfev=10000)
        
        I0_fit, V_fit, phase_scale_fit, phase_offset_fit = popt
        
        mi_fitted = signal_model(time, *popt)
        residuals = mi_signal - mi_fitted
        rmse = np.sqrt(np.mean(residuals**2))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((mi_signal - np.mean(mi_signal))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        if verbose:
            print(f"\n  Fit results:")
            print(f"    I0: {I0_fit:.6f}")
            print(f"    Visibility: {V_fit:.6f}")
            print(f"    RMSE: {rmse:.6f}")
            print(f"    R²: {r_squared:.6f}")
        
        return {
            'fitted_signal': mi_fitted,
            'residuals': residuals,
            'phase': phase_scale_fit * phase + phase_offset_fit,
            'rmse': rmse,
            'r_squared': r_squared,
            'I0': I0_fit,
            'visibility': V_fit,
            'phase_scale': phase_scale_fit,
            'phase_offset': phase_offset_fit,
            'success': True
        }
        
    except Exception as e:
        return {
            'fitted_signal': np.full_like(mi_signal, np.mean(mi_signal)),
            'residuals': mi_signal - np.mean(mi_signal),
            'rmse': np.std(mi_signal),
            'r_squared': 0.0,
            'success': False
        }


def plot_adaptive_removal_result(fit_result, 
                                 save_path='./DATA/adaptive_removal_result.png'):
    """Plot results showing which points were removed and final fit."""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    time_orig = fit_result['time_original']
    mi_orig = fit_result['mi_signal_original']
    time_kept = fit_result['time']
    mi_kept = fit_result['mi_signal']
    mi_fitted = fit_result['fitted_signal']
    
    # 1. Original signal with removed points highlighted
    ax = axes[0, 0]
    
    # Plot removed points
    if np.any(fit_result['removed_mask']):
        ax.scatter(time_orig[fit_result['removed_mask']], 
                  mi_orig[fit_result['removed_mask']], 
                  c='red', s=30, alpha=0.7, marker='x', linewidth=2, 
                  label=f'Removed ({fit_result["n_removed"]} pts)', zorder=5)
    
    # Plot kept points
    ax.scatter(time_kept, mi_kept, c='blue', s=10, alpha=0.5, 
              label=f'Kept ({len(time_kept)} pts)', zorder=3)
    
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('MI Signal', fontsize=11)
    ax.set_title(f'Outlier Removal ({fit_result["removal_fraction"]*100:.1f}% removed)', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. Final fit on kept points
    ax = axes[0, 1]
    ax.plot(time_kept, mi_kept, 'b-', linewidth=1.5, alpha=0.7, label='Data (kept)')
    ax.plot(time_kept, mi_fitted, 'r--', linewidth=1.5, alpha=0.7, label='Fitted')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('MI Signal', fontsize=11)
    ax.set_title(f'Final Fit (R² = {fit_result["r_squared"]:.6f})', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. Removal histogram (position)
    ax = axes[1, 0]
    if np.any(fit_result['removed_mask']):
        removed_indices = np.where(fit_result['removed_mask'])[0]
        ax.hist(removed_indices, bins=50, edgecolor='black', alpha=0.7, color='red')
        ax.set_xlabel('Index', fontsize=11)
        ax.set_ylabel('Number Removed', fontsize=11)
        ax.set_title('Removal Distribution by Index', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Mark edge regions
        n_total = len(time_orig)
        edge_size = n_total // 10
        ax.axvline(x=edge_size, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='10% mark')
        ax.axvline(x=n_total - edge_size, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        ax.legend(fontsize=9)
    
    # 4. Removal by time
    ax = axes[1, 1]
    if np.any(fit_result['removed_mask']):
        removed_times = time_orig[fit_result['removed_mask']]
        ax.hist(removed_times, bins=50, edgecolor='black', alpha=0.7, color='red')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Number Removed', fontsize=11)
        ax.set_title('Removal Distribution by Time', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Final residuals
    ax = axes[2, 0]
    ax.plot(time_kept, fit_result['residuals'], 'g-', linewidth=1, alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1.5)
    ax.fill_between(time_kept, fit_result['residuals'], alpha=0.3, color='green')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Residuals', fontsize=11)
    ax.set_title(f'Final Residuals (RMSE = {fit_result["rmse"]:.6f})', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 6. Residual histogram
    ax = axes[2, 1]
    ax.hist(fit_result['residuals'], bins=50, edgecolor='black', alpha=0.7, color='cyan')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual Value', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Final Residual Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Ensure the directory for the save path exists, create it if it doesn't
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nResult plot saved: {save_path}")
    
    return fig

"""
Flexible SNOM Fitter with Arbitrary Polynomial Degree
======================================================

Fits: SNOM(x) = A(x) × [offset + sin(2βx + φ₀)]

where A(x) can be any polynomial degree:
- degree=0: A(x) = A₀
- degree=1: A(x) = A₀ + A₁×x
- degree=2: A(x) = A₀ + A₁×x + A₂×x²
- degree=3: A(x) = A₀ + A₁×x + A₂×x² + A₃×x³
- etc.

You choose the degree!
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import hilbert
import warnings

def estimate_envelope(snom_signal, method='peaks', **kwargs):
    """
    Estimate the envelope A(x) from SNOM signal.
    
    Parameters:
    -----------
    snom_signal : ndarray
        SNOM signal
    method : str
        'peaks': Use peak detection (default)
        'smooth': Use Savitzky-Golay smoothing
        'polynomial': Fit polynomial to signal
    **kwargs : additional arguments for each method
        For 'smooth': window_length, polyorder
        For 'polynomial': degree
    
    Returns:
    --------
    envelope : ndarray
        Estimated envelope
    """
    from scipy.signal import find_peaks
    
    if method == 'peaks':
        # Find both peaks and troughs to get full envelope
        peaks, _ = find_peaks(snom_signal, distance=len(snom_signal)//50)
        troughs, _ = find_peaks(-snom_signal, distance=len(snom_signal)//50)
        
        if len(peaks) < 3:
            # Fall back to smooth method
            warnings.warn("Too few peaks found, using smooth method")
            return estimate_envelope(snom_signal, method='smooth')
        
        # Use peaks to estimate upper envelope, troughs for lower
        # Average gives center envelope
        from scipy.interpolate import interp1d
        
        # Upper envelope from peaks
        x_peaks = peaks
        y_peaks = snom_signal[peaks]
        if peaks[0] != 0:
            x_peaks = np.concatenate([[0], x_peaks])
            y_peaks = np.concatenate([[snom_signal[0]], y_peaks])
        if peaks[-1] != len(snom_signal)-1:
            x_peaks = np.concatenate([x_peaks, [len(snom_signal)-1]])
            y_peaks = np.concatenate([y_peaks, [snom_signal[-1]]])
        
        upper_interp = interp1d(x_peaks, y_peaks, kind='cubic', fill_value='extrapolate')
        upper_envelope = upper_interp(np.arange(len(snom_signal)))
        
        # Lower envelope from troughs
        if len(troughs) >= 3:
            x_troughs = troughs
            y_troughs = snom_signal[troughs]
            if troughs[0] != 0:
                x_troughs = np.concatenate([[0], x_troughs])
                y_troughs = np.concatenate([[snom_signal[0]], y_troughs])
            if troughs[-1] != len(snom_signal)-1:
                x_troughs = np.concatenate([x_troughs, [len(snom_signal)-1]])
                y_troughs = np.concatenate([y_troughs, [snom_signal[-1]]])
            
            lower_interp = interp1d(x_troughs, y_troughs, kind='cubic', fill_value='extrapolate')
            lower_envelope = lower_interp(np.arange(len(snom_signal)))
            
            # Average of upper and lower gives center envelope
            envelope = (upper_envelope + lower_envelope) / 2
        else:
            # Just use upper envelope
            envelope = upper_envelope
        
        # Make sure envelope is always positive
        envelope = np.maximum(envelope, np.max(np.abs(snom_signal)) * 0.01)
        
    elif method == 'smooth':
        # Smooth the signal itself
        window_length = kwargs.get('window_length', min(51, len(snom_signal)//10))
        if window_length % 2 == 0:
            window_length += 1  # Must be odd
        polyorder = kwargs.get('polyorder', 3)
        
        envelope = savgol_filter(snom_signal, window_length, polyorder)
        envelope = np.maximum(envelope, np.max(snom_signal) * 0.01)
        
    elif method == 'polynomial':
        # Fit polynomial to signal
        degree = kwargs.get('degree', 3)
        x = np.arange(len(snom_signal))
        coeffs = np.polyfit(x, snom_signal, degree)
        envelope = np.polyval(coeffs, x)
        envelope = np.maximum(envelope, np.max(snom_signal) * 0.01)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return envelope

def fit_snom(x, snom_signal, envelope_method='peaks', normalize=True, **envelope_kwargs):
    """
    Fit SNOM signal by first normalizing by envelope, then fitting.
    
    Two-step process:
    1. Estimate envelope A(x) and normalize: norm = snom / A(x)
    2. Fit: norm = offset + sin(2βx + φ₀)
    
    Parameters:
    -----------
    x : array_like
        Position array (in meters)
    snom_signal : array_like
        SNOM signal to fit
    envelope_method : str
        Method to estimate envelope:
        - 'peaks': Peak detection and interpolation (default, robust)
        - 'smooth': Savitzky-Golay smoothing
        - 'polynomial': Polynomial fit (specify degree)
        - None: No normalization, fit directly
    normalize : bool
        If True, normalize by envelope before fitting
    **envelope_kwargs : additional arguments for envelope estimation
        For 'smooth': window_length, polyorder
        For 'polynomial': degree
    
    Returns:
    --------
    fitted_signal : ndarray
        Fitted SNOM signal (in original scale)
    parameters : dict
        - 'offset': baseline offset
        - 'beta': frequency parameter (rad/m)
        - 'phi0': phase offset (rad)
    residuals : ndarray
        Fit residuals (in original scale)
    quality : dict
        - 'rmse': root mean square error
        - 'r_squared': R² coefficient
        - 'max_error': maximum absolute residual
        - 'period_um': oscillation period in micrometers
    envelope : ndarray
        Estimated envelope A(x)
    normalized : ndarray
        Normalized signal (snom / envelope)
    
    Examples:
    ---------
    >>> # Default: peak detection
    >>> fitted, params, residuals, quality, envelope, normalized = fit_snom(x, snom)
    
    >>> # Use smoothing instead
    >>> fitted, params, res, qual, env, norm = fit_snom(
    ...     x, snom, envelope_method='smooth', window_length=51
    ... )
    
    >>> # Polynomial envelope
    >>> fitted, params, res, qual, env, norm = fit_snom(
    ...     x, snom, envelope_method='polynomial', degree=3
    ... )
    
    >>> # No normalization (direct fit)
    >>> fitted, params, res, qual, env, norm = fit_snom(
    ...     x, snom, envelope_method=None
    ... )
    """
    
    x = np.asarray(x)
    snom_signal = np.asarray(snom_signal)
    
    # Validate inputs
    if len(x) != len(snom_signal):
        raise ValueError("x and snom_signal must have same length")
    if np.any(np.isnan(snom_signal)) or np.any(np.isinf(snom_signal)):
        raise ValueError("snom_signal contains NaN or Inf")
    
    # Step 1: Estimate envelope
    if normalize and envelope_method is not None:
        envelope = estimate_envelope(snom_signal, method=envelope_method, **envelope_kwargs)
        normalized = snom_signal / envelope
    else:
        # No normalization
        envelope = np.ones_like(snom_signal)
        normalized = snom_signal
    
    # Step 2: Fit simple model to normalized signal
    # Model: normalized = offset + sin(2βx + φ₀)
    def model(x, offset, beta, phi0):
        return offset + np.sin(2 * beta * x + phi0)
    
    # Estimate initial parameters
    mean_norm = np.mean(normalized)
    amplitude_norm = (np.max(normalized) - np.min(normalized)) / 2
    
    # Estimate frequency from FFT
    try:
        fft = np.fft.fft(normalized - mean_norm)
        freqs = np.fft.fftfreq(len(normalized), np.mean(np.diff(x)))
        pos_freqs = freqs[freqs > 0]
        pos_fft = np.abs(fft[freqs > 0])
        
        if len(pos_fft) > 0:
            dominant_freq = pos_freqs[np.argmax(pos_fft)]
            beta_guess = np.pi * dominant_freq
        else:
            beta_guess = np.pi / (x[-1] - x[0])
    except:
        beta_guess = np.pi / (x[-1] - x[0])
    
    # Estimate phase
    try:
        analytic = hilbert(normalized - mean_norm)
        phase_initial = np.angle(analytic[0])
    except:
        phase_initial = 0.0
    
    # Initial guess
    p0 = [mean_norm, beta_guess, phase_initial]
    
    # Fit
    try:
        popt, pcov = curve_fit(model, x, normalized, p0=p0, maxfev=10000)
    except RuntimeError as e:
        warnings.warn(f"Fitting failed: {e}")
        popt = np.array(p0)
    
    # Get fitted normalized signal
    fitted_normalized = model(x, *popt)
    
    # Scale back to original scale
    fitted_signal = fitted_normalized * envelope
    
    # Calculate residuals in original scale
    residuals = snom_signal - fitted_signal
    
    # Quality metrics
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((snom_signal - np.mean(snom_signal))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    max_error = np.max(np.abs(residuals))
    
    beta_val = popt[1]
    period_m = np.pi / np.abs(beta_val)
    period_um = period_m * 1e6
    
    # Build output
    parameters = {
        'offset': popt[0],
        'beta': popt[1],
        'phi0': popt[2]
    }
    
    quality = {
        'rmse': rmse,
        'r_squared': r_squared,
        'max_error': max_error,
        'period_um': period_um
    }
    
    return fitted_signal, parameters, residuals, quality, envelope, normalized

def plot_fit(x, snom_signal, fitted_signal, residuals, quality, parameters, 
             envelope, normalized, save_path=None):
    """
    Plot comprehensive fit results including normalization.
    
    Parameters:
    -----------
    x : ndarray
        Position array
    snom_signal : ndarray
        Original signal
    fitted_signal : ndarray
        Fitted signal
    residuals : ndarray
        Residuals
    quality : dict
        Quality metrics
    parameters : dict
        Fitted parameters
    envelope : ndarray
        Envelope A(x)
    normalized : ndarray
        Normalized signal
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    x_um = x * 1e6
    
    # 1. Original signal with envelope
    ax = axes[0, 0]
    ax.plot(x_um, snom_signal, 'b-', alpha=0.7, linewidth=1.5, label='Data')
    ax.plot(x_um, envelope, 'g--', linewidth=2, label='Envelope A(x)')
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('SNOM Signal')
    ax.set_title('Original Signal with Envelope')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Normalized signal
    ax = axes[0, 1]
    ax.plot(x_um, normalized, 'b-', alpha=0.7, linewidth=1.5, label='Normalized')
    fitted_norm = fitted_signal / envelope
    ax.plot(x_um, fitted_norm, 'r--', linewidth=2, label='Fit')
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('Normalized Signal')
    ax.set_title('Normalized Signal: SNOM/A(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Final fit vs original
    ax = axes[1, 0]
    ax.plot(x_um, snom_signal, 'b-', alpha=0.7, linewidth=1.5, label='Data')
    ax.plot(x_um, fitted_signal, 'r--', linewidth=2, label='Fit')
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('SNOM Signal')
    ax.set_title(f"Final Fit (R² = {quality['r_squared']:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Residuals
    ax = axes[1, 1]
    ax.plot(x_um, residuals, 'k-', alpha=0.7)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.fill_between(x_um, residuals, alpha=0.3)
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('Residuals')
    ax.set_title(f"RMSE = {quality['rmse']:.4e}")
    ax.grid(True, alpha=0.3)
    
    # 5. Scatter plot
    ax = axes[2, 0]
    ax.scatter(snom_signal, fitted_signal, alpha=0.5, s=20)
    lims = [min(snom_signal.min(), fitted_signal.min()),
            max(snom_signal.max(), fitted_signal.max())]
    ax.plot(lims, lims, 'r--', linewidth=2)
    ax.set_xlabel('Data')
    ax.set_ylabel('Fitted')
    ax.set_title('Actual vs Fitted')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 6. Parameters
    ax = axes[2, 1]
    ax.axis('off')
    
    text = "Parameters:\n" + "="*30 + "\n"
    for key, val in parameters.items():
        text += f"{key:8s} = {val:.6e}\n"
    
    text += "\nQuality:\n" + "="*30 + "\n"
    text += f"RMSE     = {quality['rmse']:.6e}\n"
    text += f"R²       = {quality['r_squared']:.6f}\n"
    text += f"Max err  = {quality['max_error']:.6e}\n"
    text += f"Period   = {quality['period_um']:.4f} μm\n"
    
    text += "\nModel:\n" + "="*30 + "\n"
    text += "SNOM = A(x) × [offset + sin(2βx+φ₀)]\n"
    text += "where A(x) = envelope"
    
    ax.text(0.1, 0.9, text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

def compare_degrees(x, snom_signal, degrees=[0, 1, 2, 3], save_path=None):
    """
    Compare fits with different polynomial degrees.
    
    Parameters:
    -----------
    x : ndarray
        Position array
    snom_signal : ndarray
        SNOM signal
    degrees : list of int
        Degrees to test
    save_path : str, optional
        Path to save comparison plot
        
    Returns:
    --------
    results : dict
        Results for each degree
    """
    import matplotlib.pyplot as plt
    
    results = {}
    
    print(f"\nComparing polynomial degrees: {degrees}")
    print("="*70)
    
    for deg in degrees:
        fitted, params, residuals, quality = fit_snom(x, snom_signal, degree=deg)
        results[deg] = {
            'fitted': fitted,
            'params': params,
            'residuals': residuals,
            'quality': quality
        }
        print(f"Degree {deg}: R² = {quality['r_squared']:.6f}, RMSE = {quality['rmse']:.6e}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x_um = x * 1e6
    
    # Fits comparison
    ax = axes[0, 0]
    ax.plot(x_um, snom_signal, 'k-', alpha=0.5, linewidth=1, label='Data')
    colors = ['r', 'g', 'b', 'm', 'c', 'y']
    for i, deg in enumerate(degrees):
        ax.plot(x_um, results[deg]['fitted'], '--', linewidth=2,
               color=colors[i % len(colors)],
               label=f"Degree {deg}")
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('SNOM Signal')
    ax.set_title('Fits Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Residuals comparison
    ax = axes[0, 1]
    for i, deg in enumerate(degrees):
        ax.plot(x_um, results[deg]['residuals'], alpha=0.7, linewidth=1,
               color=colors[i % len(colors)], label=f"Degree {deg}")
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # R² comparison
    ax = axes[1, 0]
    r2_values = [results[deg]['quality']['r_squared'] for deg in degrees]
    ax.bar(degrees, r2_values, color='skyblue', edgecolor='black')
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('R²')
    ax.set_title('R² vs Degree')
    ax.set_xticks(degrees)
    ax.grid(True, alpha=0.3, axis='y')
    
    # RMSE comparison
    ax = axes[1, 1]
    rmse_values = [results[deg]['quality']['rmse'] for deg in degrees]
    ax.bar(degrees, rmse_values, color='salmon', edgecolor='black')
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE vs Degree')
    ax.set_xticks(degrees)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved comparison: {save_path}")
    
    plt.show()
    
    return results

# Test
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTING AGGRESSIVE EDGE REMOVAL")
    print("=" * 70)
    
    # Generate test signal
    np.random.seed(42)
    wavelength = 632.8e-9
    
    time = np.linspace(0, 0.2, 2000)
    piezo = 1.5e-6 * (time / 0.2)
    piezo += 0.15 * 1.5e-6 * np.sin(10 * np.pi * time)
    
    phase = 4 * np.pi * piezo / wavelength
    mi_signal = 1 + 0.9 * np.cos(phase)
    mi_signal += 0.01 * np.random.randn(len(time))
    
    # Add artificial edge effects
    edge_n = 50
    mi_signal[:edge_n] += np.linspace(0.1, 0, edge_n)
    mi_signal[-edge_n:] += np.linspace(0, 0.1, edge_n)
    
    print("\nTest signal with artificial edge effects created")
    
    # Try different discard fractions
    for frac in [0.03, 0.05, 0.10]:
        print("\n" + "=" * 70)
        print(f"Testing with edge_discard_fraction = {frac}")
        
        fit = fit_with_aggressive_edge_removal(time, mi_signal, 
                                               edge_discard_fraction=frac,
                                               verbose=True)
        
        if fit['success']:
            plot_before_after_edge_removal(time, mi_signal, fit,
                                          save_path=f'/mnt/user-data/outputs/edge_removal_{int(frac*100)}pct.png')
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print("For your data with stubborn left edge effect:")
    print("  Start with edge_discard_fraction = 0.05 (5% from each side)")
    print("  If edge still visible, increase to 0.10 (10% from each side)")
    print("  Check residual plots to verify uniformity")


# Test
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EDGE-EFFECT-FREE FITTER TEST")
    print("=" * 70)
    
    # Generate test signal with edge effects
    np.random.seed(42)
    wavelength = 632.8e-9
    
    time = np.linspace(0, 0.2, 2000)
    piezo = 1.5e-6 * (time / 0.2)
    piezo += 0.15 * 1.5e-6 * np.sin(10 * np.pi * time)
    
    phase = 4 * np.pi * piezo / wavelength
    mi_signal = 1 + 0.9 * np.cos(phase)
    mi_signal += 0.01 * np.random.randn(len(time))
    
    # Test with and without edge trimming
    print("\n" + "="*70)
    print("TEST 1: No edge trimming, with padding")
    fit1 = fit_mi_signal_no_edges(time, mi_signal, edge_trim=0.0, use_padding=True)
    
    print("\n" + "="*70)
    print("TEST 2: With 5% edge trimming and padding")
    fit2 = fit_mi_signal_no_edges(time, mi_signal, edge_trim=0.05, use_padding=True)
    
    # Plot
    plot_edge_analysis(fit2['time'], mi_signal if fit2['time'] is time else 
                      mi_signal[int(len(mi_signal)*0.05):-int(len(mi_signal)*0.05)], 
                      fit2)
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION:")
    print("=" * 70)
    edge_ratio1 = max(fit1['left_edge_rmse'], fit1['right_edge_rmse']) / fit1['center_rmse']
    edge_ratio2 = max(fit2['left_edge_rmse'], fit2['right_edge_rmse']) / fit2['center_rmse']
    
    print(f"Without edge trim: Edge/Center ratio = {edge_ratio1:.2f}")
    print(f"With 5% edge trim: Edge/Center ratio = {edge_ratio2:.2f}")
    
    if edge_ratio2 < edge_ratio1:
        print("\n✓ Edge trimming helps! Use edge_trim=0.05 or higher")
    else:
        print("\n→ Padding alone is sufficient")
