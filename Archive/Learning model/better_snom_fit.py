"""
Better SNOM Fitter
==================

Two approaches:
1. You provide envelope A(x) → we just fit oscillation
2. We fit everything together with proper constraints

Much more reliable!
"""

import numpy as np
from scipy.optimize import curve_fit, differential_evolution
from scipy.signal import hilbert
import warnings


def fit_snom_with_envelope(x, snom_signal, envelope):
    """
    Fit SNOM when you already know the envelope.
    
    Given: envelope A(x)
    Fit: normalized = snom / A(x) = offset + sin(2βx + φ₀)
    
    Parameters:
    -----------
    x : array_like
        Position (meters)
    snom_signal : array_like
        SNOM signal
    envelope : array_like
        Known envelope A(x) (same length as snom_signal)
    
    Returns:
    --------
    fitted_signal : ndarray
    parameters : dict with 'offset', 'beta', 'phi0'
    residuals : ndarray
    quality : dict
    normalized : ndarray
    """
    x = np.asarray(x)
    snom_signal = np.asarray(snom_signal)
    envelope = np.asarray(envelope)
    
    # Normalize
    normalized = snom_signal / envelope
    
    # Fit simple model
    def model(x, offset, beta, phi0):
        return offset + np.sin(2 * beta * x + phi0)
    
    # Initial guess
    mean_norm = np.mean(normalized)
    
    # Estimate beta from FFT
    try:
        fft = np.fft.fft(normalized - mean_norm)
        freqs = np.fft.fftfreq(len(normalized), np.mean(np.diff(x)))
        pos_freqs = freqs[freqs > 0]
        pos_fft = np.abs(fft[freqs > 0])
        if len(pos_fft) > 0:
            beta_guess = np.pi * pos_freqs[np.argmax(pos_fft)]
        else:
            beta_guess = np.pi / (x[-1] - x[0])
    except:
        beta_guess = np.pi / (x[-1] - x[0])
    
    # Estimate phase
    try:
        analytic = hilbert(normalized - mean_norm)
        phi0_guess = np.angle(analytic[0])
    except:
        phi0_guess = 0.0
    
    p0 = [mean_norm, beta_guess, phi0_guess]
    
    # Fit
    try:
        popt, _ = curve_fit(model, x, normalized, p0=p0, maxfev=10000)
    except:
        warnings.warn("Fit failed, using initial guess")
        popt = p0
    
    fitted_normalized = model(x, *popt)
    fitted_signal = fitted_normalized * envelope
    
    residuals = snom_signal - fitted_signal
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((snom_signal - np.mean(snom_signal))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    parameters = {
        'offset': popt[0],
        'beta': popt[1],
        'phi0': popt[2]
    }
    
    quality = {
        'rmse': rmse,
        'r_squared': r_squared,
        'max_error': np.max(np.abs(residuals)),
        'period_um': (np.pi / np.abs(popt[1])) * 1e6
    }
    
    return fitted_signal, parameters, residuals, quality, normalized


def fit_snom_full(x, snom_signal, envelope_degree=2, method='robust'):
    """
    Fit both envelope and oscillation together.
    
    Model: SNOM(x) = A(x) × [offset + sin(2βx + φ₀)]
    where A(x) = A₀ + A₁×x + A₂×x² + ... (polynomial envelope)
    
    Parameters:
    -----------
    x : array_like
        Position (meters)
    snom_signal : array_like
        SNOM signal
    envelope_degree : int
        Polynomial degree for envelope (default: 2)
    method : str
        'fast': curve_fit (may fail)
        'robust': differential_evolution (slower but more reliable)
    
    Returns:
    --------
    fitted_signal : ndarray
    parameters : dict
        Envelope: 'A0', 'A1', ... 'A{degree}'
        Oscillation: 'offset', 'beta', 'phi0'
    residuals : ndarray
    quality : dict
    envelope : ndarray
    normalized : ndarray
    """
    x = np.asarray(x)
    snom_signal = np.asarray(snom_signal)
    
    n_env_params = envelope_degree + 1
    
    def model(x, *params):
        # Envelope coefficients
        env_coeffs = params[:n_env_params]
        envelope = np.zeros_like(x)
        for i, coeff in enumerate(env_coeffs):
            envelope += coeff * (x ** i)
        
        # Oscillation parameters
        offset = params[n_env_params]
        beta = params[n_env_params + 1]
        phi0 = params[n_env_params + 2]
        
        oscillation = offset + np.sin(2 * beta * x + phi0)
        return envelope * oscillation
    
    # Initial guess
    mean_signal = np.mean(snom_signal)
    
    # Estimate envelope from polynomial fit
    try:
        env_poly = np.polyfit(x, snom_signal, envelope_degree)
        env_coeffs_guess = env_poly[::-1]  # Reverse to [A0, A1, A2, ...]
    except:
        env_coeffs_guess = [mean_signal] + [0.0] * envelope_degree
    
    # Estimate oscillation parameters
    try:
        fft = np.fft.fft(snom_signal - mean_signal)
        freqs = np.fft.fftfreq(len(snom_signal), np.mean(np.diff(x)))
        pos_freqs = freqs[freqs > 0]
        pos_fft = np.abs(fft[freqs > 0])
        if len(pos_fft) > 0:
            beta_guess = np.pi * pos_freqs[np.argmax(pos_fft)]
        else:
            beta_guess = np.pi / (x[-1] - x[0])
    except:
        beta_guess = np.pi / (x[-1] - x[0])
    
    p0 = list(env_coeffs_guess) + [1.0, beta_guess, 0.0]
    
    if method == 'robust':
        # Use differential evolution for robust fitting
        def objective(params):
            predicted = model(x, *params)
            return np.sum((snom_signal - predicted)**2)
        
        # Build bounds
        bounds = []
        # Envelope bounds (allow wide range)
        bounds.append((mean_signal * 0.01, mean_signal * 10))  # A0
        for i in range(envelope_degree):
            bounds.append((-abs(env_coeffs_guess[i+1])*100, abs(env_coeffs_guess[i+1])*100))
        
        # Oscillation bounds
        bounds.append((0.1, 10.0))  # offset
        bounds.append((beta_guess * 0.1, beta_guess * 10))  # beta
        bounds.append((-2*np.pi, 2*np.pi))  # phi0
        
        print("Fitting with robust method (may take a minute)...")
        result = differential_evolution(objective, bounds, maxiter=500, seed=42, disp=False)
        popt = result.x
        
    else:  # 'fast'
        try:
            popt, _ = curve_fit(model, x, snom_signal, p0=p0, maxfev=10000)
        except:
            warnings.warn("Fast fit failed, trying robust method...")
            return fit_snom_full(x, snom_signal, envelope_degree, method='robust')
    
    # Calculate results
    fitted_signal = model(x, *popt)
    
    # Extract envelope
    env_coeffs = popt[:n_env_params]
    envelope = np.zeros_like(x)
    for i, coeff in enumerate(env_coeffs):
        envelope += coeff * (x ** i)
    
    normalized = snom_signal / envelope
    
    residuals = snom_signal - fitted_signal
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((snom_signal - np.mean(snom_signal))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Build parameter dict
    parameters = {}
    for i in range(n_env_params):
        parameters[f'A{i}'] = popt[i]
    parameters['offset'] = popt[n_env_params]
    parameters['beta'] = popt[n_env_params + 1]
    parameters['phi0'] = popt[n_env_params + 2]
    
    quality = {
        'rmse': rmse,
        'r_squared': r_squared,
        'max_error': np.max(np.abs(residuals)),
        'period_um': (np.pi / np.abs(popt[n_env_params + 1])) * 1e6,
        'envelope_degree': envelope_degree
    }
    
    return fitted_signal, parameters, residuals, quality, envelope, normalized


def plot_fit(x, snom_signal, fitted_signal, residuals, quality, parameters,
             envelope=None, normalized=None, save_path=None):
    """Plot fit results"""
    import matplotlib.pyplot as plt
    
    has_components = envelope is not None and normalized is not None
    
    if has_components:
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    x_um = x * 1e6
    
    if has_components:
        # With envelope and normalized
        # 1. Original with envelope
        ax = axes[0, 0]
        ax.plot(x_um, snom_signal, 'b-', alpha=0.7, linewidth=1.5, label='Data')
        ax.plot(x_um, envelope, 'g--', linewidth=2, label='Envelope')
        ax.set_xlabel('Position (μm)')
        ax.set_ylabel('SNOM Signal')
        ax.set_title('Original Signal with Envelope')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Normalized
        ax = axes[0, 1]
        ax.plot(x_um, normalized, 'b-', alpha=0.7, linewidth=1.5, label='Normalized')
        if fitted_signal is not None:
            fitted_norm = fitted_signal / envelope
            ax.plot(x_um, fitted_norm, 'r--', linewidth=2, label='Fit')
        ax.set_xlabel('Position (μm)')
        ax.set_ylabel('Normalized')
        ax.set_title('Normalized Signal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax_row = 1
    else:
        ax_row = 0
    
    # Original vs fit
    ax = axes[ax_row, 0]
    ax.plot(x_um, snom_signal, 'b-', alpha=0.7, linewidth=1.5, label='Data')
    ax.plot(x_um, fitted_signal, 'r--', linewidth=2, label='Fit')
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('SNOM Signal')
    ax.set_title(f"R² = {quality['r_squared']:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Residuals
    ax = axes[ax_row, 1]
    ax.plot(x_um, residuals, 'k-', alpha=0.7)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.fill_between(x_um, residuals, alpha=0.3)
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('Residuals')
    ax.set_title(f"RMSE = {quality['rmse']:.4e}")
    ax.grid(True, alpha=0.3)
    
    # Scatter
    ax = axes[ax_row+1, 0]
    ax.scatter(snom_signal, fitted_signal, alpha=0.5, s=20)
    lims = [min(snom_signal.min(), fitted_signal.min()),
            max(snom_signal.max(), fitted_signal.max())]
    ax.plot(lims, lims, 'r--', linewidth=2)
    ax.set_xlabel('Data')
    ax.set_ylabel('Fitted')
    ax.set_title('Actual vs Fitted')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Parameters
    ax = axes[ax_row+1, 1]
    ax.axis('off')
    
    text = "Parameters:\n" + "="*30 + "\n"
    for key, val in parameters.items():
        text += f"{key:8s} = {val:.6e}\n"
    text += "\nQuality:\n" + "="*30 + "\n"
    text += f"RMSE     = {quality['rmse']:.6e}\n"
    text += f"R²       = {quality['r_squared']:.6f}\n"
    text += f"Max err  = {quality['max_error']:.6e}\n"
    text += f"Period   = {quality['period_um']:.4f} μm"
    
    ax.text(0.1, 0.9, text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# Test
if __name__ == "__main__":
    print("="*70)
    print("Better SNOM Fitter - Test")
    print("="*70)
    
    # Generate test data
    np.random.seed(42)
    x = np.linspace(0, 20e-6, 2000)
    
    # Quadratic envelope
    A0, A1, A2 = 0.1, 4e-6, 1e-10
    envelope_true = A0 + A1*x + A2*x**2
    
    # Oscillation
    offset_true = 1.0
    beta_true = 1e6
    phi0_true = 0.5
    
    normalized_true = offset_true + np.sin(2*beta_true*x + phi0_true)
    snom_true = envelope_true * normalized_true
    snom_signal = snom_true + 0.01 * np.random.randn(len(x))
    
    print(f"\nTest data: {len(x)} points over {x[-1]*1e6:.1f} μm")
    print(f"True envelope: A₀={A0}, A₁={A1:.2e}, A₂={A2:.2e}")
    print(f"True period: {np.pi/beta_true*1e6:.4f} μm")
    
    # Test 1: Fit with known envelope
    print("\n" + "="*70)
    print("TEST 1: Fit with KNOWN envelope")
    print("="*70)
    
    fitted, params, res, qual, norm = fit_snom_with_envelope(
        x, snom_signal, envelope_true
    )
    
    print(f"R² = {qual['r_squared']:.6f}")
    print(f"RMSE = {qual['rmse']:.6e}")
    print(f"Period = {qual['period_um']:.4f} μm")
    
    plot_fit(x, snom_signal, fitted, res, qual, params,
            envelope_true, norm,
            save_path='/mnt/user-data/outputs/better_snom_known_envelope.png')
    
    # Test 2: Fit everything together
    print("\n" + "="*70)
    print("TEST 2: Fit BOTH envelope and oscillation (robust)")
    print("="*70)
    
    fitted2, params2, res2, qual2, env2, norm2 = fit_snom_full(
        x, snom_signal,
        envelope_degree=2,
        method='robust'
    )
    
    print(f"R² = {qual2['r_squared']:.6f}")
    print(f"RMSE = {qual2['rmse']:.6e}")
    print(f"Period = {qual2['period_um']:.4f} μm")
    print(f"\nFitted envelope: A₀={params2['A0']:.6e}, A₁={params2['A1']:.6e}, A₂={params2['A2']:.6e}")
    print(f"True envelope:   A₀={A0:.6e}, A₁={A1:.6e}, A₂={A2:.6e}")
    
    plot_fit(x, snom_signal, fitted2, res2, qual2, params2,
            env2, norm2,
            save_path='/mnt/user-data/outputs/better_snom_full_fit.png')
    
    print("\n" + "="*70)
    print("✓ Test complete!")
    print("="*70)
