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
from matplotlib import pyplot as plt
import os


def estimate_envelope_hilbert(snom_signal, x, smooth_method='polynomial', **kwargs):
    """
    Estimate envelope using Hilbert transform.
    
    The Hilbert transform gives instantaneous amplitude:
    envelope = |hilbert(signal)|
    
    Parameters:
    -----------
    snom_signal : ndarray
        SNOM signal
    x : ndarray
        Position array (for polynomial fitting)
    smooth_method : str
        'polynomial': Fit polynomial to Hilbert envelope (best!)
        'savgol': Savitzky-Golay smoothing
        'none': Raw Hilbert envelope (has ripples)
    **kwargs : additional arguments
        For 'polynomial': degree (default: 2)
        For 'savgol': window_length, polyorder
    
    Returns:
    --------
    envelope : ndarray
        Estimated envelope
    """
    # Get Hilbert envelope
    analytic_signal = hilbert(snom_signal)
    envelope_raw = np.abs(analytic_signal)
    
    if smooth_method == 'polynomial':
        # Fit polynomial to smooth the envelope
        degree = kwargs.get('degree', 2)
        poly_coeffs = np.polyfit(np.arange(len(snom_signal)), envelope_raw, degree)
        envelope = np.polyval(poly_coeffs, np.arange(len(snom_signal)))
        
    elif smooth_method == 'savgol':
        # Savitzky-Golay smoothing
        from scipy.signal import savgol_filter
        window_length = kwargs.get('window_length', min(51, len(snom_signal)//10))
        if window_length % 2 == 0:
            window_length += 1
        polyorder = kwargs.get('polyorder', 3)
        envelope = savgol_filter(envelope_raw, window_length, polyorder)
        
    elif smooth_method == 'none':
        # Raw Hilbert envelope (will have ripples)
        envelope = envelope_raw
        
    else:
        raise ValueError(f"Unknown smooth_method: {smooth_method}")
    
    # Ensure envelope is positive
    envelope = np.maximum(envelope, np.max(snom_signal) * 0.01)
    
    return envelope




def fit_snom_hilbert(x, snom_signal, trim_edges=0.0, envelope_degree=2):
    """
    Fit SNOM using Hilbert transform to extract envelope automatically.
    
    This is the SIMPLEST method:
    1. Use Hilbert transform to get envelope
    2. Fit polynomial to smooth it
    3. Normalize and fit oscillation
    
    Parameters:
    -----------
    x : array_like
        Position (meters)
    snom_signal : array_like
        SNOM signal
    trim_edges : float
        Fraction to trim from each edge (0.0 - 0.5)
    envelope_degree : int
        Polynomial degree for envelope smoothing (default: 2)
    
    Returns:
    --------
    fitted_signal : ndarray
    parameters : dict with 'offset', 'beta', 'phi0'
    residuals : ndarray
    quality : dict
    envelope : ndarray
    normalized : ndarray
    
    Example:
    --------
    >>> # Automatic envelope extraction with Hilbert!
    >>> fitted, params, res, qual, env, norm = fit_snom_hilbert(
    ...     x, snom_signal, trim_edges=0.1
    ... )
    """
    x = np.asarray(x)
    snom_signal = np.asarray(snom_signal)
    
    print(f"Using Hilbert transform with degree-{envelope_degree} polynomial smoothing...")
    
    # Extract envelope using Hilbert transform
    envelope = estimate_envelope_hilbert(
        snom_signal, x,
        smooth_method='polynomial',
        degree=envelope_degree
    )
    
    # Now fit using the extracted envelope
    fitted, params, residuals, quality, normalized = fit_snom_with_envelope(
        x, snom_signal, envelope, trim_edges=trim_edges
    )
    
    # Return with envelope included
    return fitted, params, residuals, quality, envelope, normalized


def fit_snom_with_envelope(x, snom_signal, envelope, trim_edges=0.0):
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
    trim_edges : float
        Fraction of data to trim from each edge (0.0 - 0.5)
        Example: 0.1 = trim 10% from start and 10% from end
    
    Returns:
    --------
    fitted_signal : ndarray (full length)
    parameters : dict with 'offset', 'beta', 'phi0'
    residuals : ndarray (full length)
    quality : dict
    normalized : ndarray (full length)
    """
    x = np.asarray(x)
    snom_signal = np.asarray(snom_signal)
    envelope = np.asarray(envelope)
    
    # Trim edges if requested
    if trim_edges > 0:
        n = len(x)
        trim_n = int(n * trim_edges)
        trim_slice = slice(trim_n, n - trim_n)
        
        x_trim = x[trim_slice]
        snom_trim = snom_signal[trim_slice]
        envelope_trim = envelope[trim_slice]
    else:
        x_trim = x
        snom_trim = snom_signal
        envelope_trim = envelope
    
    # Normalize (use trimmed data)
    normalized_trim = snom_trim / envelope_trim
    
    # Fit simple model on trimmed data
    def model(x, offset, beta, phi0):
        return offset + np.sin(2 * beta * x + phi0)
    
    # Initial guess
    mean_norm = np.mean(normalized_trim)
    
    # Estimate beta from FFT
    try:
        fft = np.fft.fft(normalized_trim - mean_norm)
        freqs = np.fft.fftfreq(len(normalized_trim), np.mean(np.diff(x_trim)))
        pos_freqs = freqs[freqs > 0]
        pos_fft = np.abs(fft[freqs > 0])
        if len(pos_fft) > 0:
            beta_guess = np.pi * pos_freqs[np.argmax(pos_fft)]
        else:
            beta_guess = np.pi / (x_trim[-1] - x_trim[0])
    except:
        beta_guess = np.pi / (x_trim[-1] - x_trim[0])
    
    # Estimate phase
    try:
        analytic = hilbert(normalized_trim - mean_norm)
        phi0_guess = np.angle(analytic[0])
    except:
        phi0_guess = 0.0
    
    p0 = [mean_norm, beta_guess, phi0_guess]
    
    # Fit on trimmed data
    try:
        popt, _ = curve_fit(model, x_trim, normalized_trim, p0=p0, maxfev=10000)
    except:
        warnings.warn("Fit failed, using initial guess")
        popt = p0
    
    # Apply fitted model to FULL data
    normalized = snom_signal / envelope  # Full normalized signal
    fitted_normalized = model(x, *popt)  # Fitted model on full x
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
        'period_um': (np.pi / np.abs(popt[1])) * 1e6,
        'trimmed': trim_edges > 0,
        'trim_fraction': trim_edges
    }
    
    return fitted_signal, parameters, residuals, quality, normalized


def fit_snom_full(x, snom_signal, envelope_degree=2, method='robust', trim_edges=0.0):
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
    trim_edges : float
        Fraction of data to trim from each edge (0.0 - 0.5)
        Example: 0.1 = trim 10% from start and 10% from end
    
    Returns:
    --------
    fitted_signal : ndarray (full length)
    parameters : dict
        Envelope: 'A0', 'A1', ... 'A{degree}'
        Oscillation: 'offset', 'beta', 'phi0'
    residuals : ndarray (full length)
    quality : dict
    envelope : ndarray (full length)
    normalized : ndarray (full length)
    """
    x = np.asarray(x)
    snom_signal = np.asarray(snom_signal)
    
    # Trim edges if requested
    if trim_edges > 0:
        n = len(x)
        trim_n = int(n * trim_edges)
        trim_slice = slice(trim_n, n - trim_n)
        
        x_trim = x[trim_slice]
        snom_trim = snom_signal[trim_slice]
        print(f"Trimming {trim_edges*100:.0f}% from each edge ({trim_n} points)")
    else:
        x_trim = x
        snom_trim = snom_signal
    
    n_env_params = envelope_degree + 1
    
    def model(x_data, *params):
        # Envelope coefficients
        env_coeffs = params[:n_env_params]
        envelope_data = np.zeros_like(x_data)
        for i, coeff in enumerate(env_coeffs):
            envelope_data += coeff * (x_data ** i)
        
        # Oscillation parameters
        offset = params[n_env_params]
        beta = params[n_env_params + 1]
        phi0 = params[n_env_params + 2]
        
        oscillation = offset + np.sin(2 * beta * x_data + phi0)
        return envelope_data * oscillation
    
    # Initial guess (on trimmed data)
    mean_signal = np.mean(snom_trim)
    
    # Estimate envelope from polynomial fit
    try:
        env_poly = np.polyfit(x_trim, snom_trim, envelope_degree)
        env_coeffs_guess = env_poly[::-1]  # Reverse to [A0, A1, A2, ...]
    except:
        env_coeffs_guess = [mean_signal] + [0.0] * envelope_degree
    
    # Estimate oscillation parameters
    try:
        fft = np.fft.fft(snom_trim - mean_signal)
        freqs = np.fft.fftfreq(len(snom_trim), np.mean(np.diff(x_trim)))
        pos_freqs = freqs[freqs > 0]
        pos_fft = np.abs(fft[freqs > 0])
        if len(pos_fft) > 0:
            beta_guess = np.pi * pos_freqs[np.argmax(pos_fft)]
        else:
            beta_guess = np.pi / (x_trim[-1] - x_trim[0])
    except:
        beta_guess = np.pi / (x_trim[-1] - x_trim[0])
    
    p0 = list(env_coeffs_guess) + [1.0, beta_guess, 0.0]
    
    if method == 'robust':
        # Use differential evolution for robust fitting (on trimmed data)
        def objective(params):
            predicted = model(x_trim, *params)
            return np.sum((snom_trim - predicted)**2)
        
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
            popt, _ = curve_fit(model, x_trim, snom_trim, p0=p0, maxfev=10000)
        except:
            warnings.warn("Fast fit failed, trying robust method...")
            return fit_snom_full(x, snom_signal, envelope_degree, method='robust', trim_edges=trim_edges)
    
    # Calculate results on FULL data
    fitted_signal = model(x, *popt)
    
    # Extract envelope on full data
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
        'envelope_degree': envelope_degree,
        'trimmed': trim_edges > 0,
        'trim_fraction': trim_edges
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

def fit_snom_ultimate(x, snom_signal, trim_edges=0.1, verbose=True):
    """
    Ultimate SNOM fitter that handles everything.
    
    Steps:
    1. Extract envelope with Hilbert
    2. Normalize
    3. Fit normalized signal with BOTH offset(x) and amplitude(x) trends
    4. Combine everything
    
    Parameters:
    -----------
    x : array_like
        Position (meters)
    snom_signal : array_like
        SNOM signal
    trim_edges : float
        Fraction to trim from each edge
    verbose : bool
        Print progress
    
    Returns:
    --------
    fitted_signal : ndarray
    parameters : dict
    residuals : ndarray
    quality : dict
    components : dict with 'envelope', 'normalized', 'offset', 'amplitude'
    """
    x = np.asarray(x)
    snom_signal = np.asarray(snom_signal)
    
    if verbose:
        print("="*70)
        print("Ultimate SNOM Fitter")
        print("="*70)
    
    # Trim if requested
    if trim_edges > 0:
        n = len(x)
        trim_n = int(n * trim_edges)
        trim_slice = slice(trim_n, n - trim_n)
        x_trim = x[trim_slice]
        snom_trim = snom_signal[trim_slice]
        if verbose:
            print(f"Trimming {trim_edges*100:.0f}% from each edge ({trim_n} points)")
    else:
        x_trim = x
        snom_trim = snom_signal
    
    # Step 1: Extract envelope with Hilbert
    if verbose:
        print("\nStep 1: Extracting envelope with Hilbert transform...")
    
    analytic = hilbert(snom_signal)
    envelope_raw = np.abs(analytic)
    
    # Smooth envelope with polynomial
    poly_env = np.polyfit(np.arange(len(snom_signal)), envelope_raw, 3)
    envelope = np.polyval(poly_env, np.arange(len(snom_signal)))
    envelope = np.maximum(envelope, np.max(snom_signal) * 0.01)
    
    if verbose:
        print(f"  Envelope range: {envelope.min():.4f} to {envelope.max():.4f}")
    
    # Step 2: Normalize
    normalized = snom_signal / envelope
    normalized_trim = snom_trim / envelope[trim_slice if trim_edges > 0 else slice(None)]
    
    if verbose:
        print(f"\nStep 2: Normalized signal")
        print(f"  Range: {normalized.min():.4f} to {normalized.max():.4f}")
    
    # Step 3: Fit normalized signal with trends
    if verbose:
        print("\nStep 3: Fitting normalized signal with offset & amplitude trends...")
    
    # Model: norm(x) = offset(x) + amplitude(x) × sin(2βx + φ₀)
    # where offset(x) and amplitude(x) are polynomials
    
    def normalized_model(x_data, offset0, offset1, amp0, amp1, beta, phi0):
        """
        Normalized model with linear trends:
        norm = [offset0 + offset1*x] + [amp0 + amp1*x] * sin(2*beta*x + phi0)
        """
        offset_x = offset0 + offset1 * x_data
        amp_x = amp0 + amp1 * x_data
        return offset_x + amp_x * np.sin(2 * beta * x_data + phi0)
    
    # Estimate initial parameters
    mean_norm = np.mean(normalized_trim)
    amplitude_norm = (np.max(normalized_trim) - np.min(normalized_trim)) / 2
    
    # Estimate trends
    offset_trend = np.polyfit(x_trim, normalized_trim, 1)
    offset1_guess = offset_trend[0]
    offset0_guess = offset_trend[1]
    
    # Estimate beta from FFT
    try:
        fft = np.fft.fft(normalized_trim - np.mean(normalized_trim))
        freqs = np.fft.fftfreq(len(normalized_trim), np.mean(np.diff(x_trim)))
        pos_freqs = freqs[freqs > 0]
        pos_fft = np.abs(fft[freqs > 0])
        if len(pos_fft) > 0:
            beta_guess = np.pi * pos_freqs[np.argmax(pos_fft)]
        else:
            beta_guess = np.pi / (x_trim[-1] - x_trim[0])
    except:
        beta_guess = np.pi / (x_trim[-1] - x_trim[0])
    
    # Estimate phase
    try:
        analytic_norm = hilbert(normalized_trim - mean_norm)
        phi0_guess = np.angle(analytic_norm[0])
    except:
        phi0_guess = 0.0
    
    p0 = [offset0_guess, offset1_guess, amplitude_norm, 0.0, beta_guess, phi0_guess]
    
    if verbose:
        print(f"  Initial beta: {beta_guess:.4e} rad/m")
        print(f"  Initial offset: {offset0_guess:.4f} + {offset1_guess:.4e}×x")
    
    # Fit
    try:
        popt, _ = curve_fit(normalized_model, x_trim, normalized_trim, p0=p0, 
                           maxfev=10000)
    except:
        warnings.warn("Fit failed, using initial guess")
        popt = p0
    
    # Apply to full data
    fitted_normalized = normalized_model(x, *popt)
    fitted_signal = fitted_normalized * envelope
    
    # Extract components
    offset_x = popt[0] + popt[1] * x
    amp_x = popt[2] + popt[3] * x
    
    residuals = snom_signal - fitted_signal
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((snom_signal - np.mean(snom_signal))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    parameters = {
        'offset0': popt[0],
        'offset1': popt[1],
        'amp0': popt[2],
        'amp1': popt[3],
        'beta': popt[4],
        'phi0': popt[5]
    }
    
    quality = {
        'rmse': rmse,
        'r_squared': r_squared,
        'max_error': np.max(np.abs(residuals)),
        'period_um': (np.pi / np.abs(popt[4])) * 1e6
    }
    
    components = {
        'envelope': envelope,
        'normalized': normalized,
        'fitted_normalized': fitted_normalized,
        'offset': offset_x,
        'amplitude': amp_x
    }
    
    if verbose:
        print(f"\nResults:")
        print(f"  R² = {r_squared:.6f}")
        print(f"  RMSE = {rmse:.6e}")
        print(f"  Period = {quality['period_um']:.4f} μm")
    
    return fitted_signal, parameters, residuals, quality, components

def plot_ultimate_fit(x, snom_signal, fitted_signal, residuals, quality, 
                     parameters, components, save_path=0):
    """Plot comprehensive results"""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    x_um = x * 1e6
    
    # 1. Original with envelope
    ax = axes[0, 0]
    ax.plot(x_um, snom_signal, 'b-', alpha=0.7, linewidth=1.5, label='Data')
    ax.plot(x_um, components['envelope'], 'g--', linewidth=2, label='Envelope')
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('SNOM Signal')
    ax.set_title('Original Signal with Envelope')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Normalized signal
    ax = axes[0, 1]
    ax.plot(x_um, components['normalized'], 'b-', alpha=0.7, linewidth=1.5, label='Normalized')
    ax.plot(x_um, components['fitted_normalized'], 'r--', linewidth=2, label='Fit')
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('Normalized')
    ax.set_title('Normalized Signal with Trends')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Offset trend
    ax = axes[0, 2]
    ax.plot(x_um, components['offset'], 'g-', linewidth=2)
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('Offset(x)')
    ax.set_title('Varying Offset')
    ax.grid(True, alpha=0.3)
    
    # 4. Amplitude trend
    ax = axes[1, 0]
    ax.plot(x_um, components['amplitude'], 'm-', linewidth=2)
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('Amplitude(x)')
    ax.set_title('Varying Amplitude')
    ax.grid(True, alpha=0.3)
    
    # 5. Final fit
    ax = axes[1, 1]
    ax.plot(x_um, snom_signal, 'b-', alpha=0.7, linewidth=1.5, label='Data')
    ax.plot(x_um, fitted_signal, 'r--', linewidth=2, label='Fit')
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('SNOM Signal')
    ax.set_title(f"Final Fit (R² = {quality['r_squared']:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Residuals
    ax = axes[1, 2]
    ax.plot(x_um, residuals, 'k-', alpha=0.7)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.fill_between(x_um, residuals, alpha=0.3)
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('Residuals')
    ax.set_title(f"RMSE = {quality['rmse']:.4e}")
    ax.grid(True, alpha=0.3)
    
    # 7. Scatter
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
    
    # 8. Normalized residuals
    ax = axes[2, 1]
    norm_residuals = components['normalized'] - components['fitted_normalized']
    ax.plot(x_um, norm_residuals, 'k-', alpha=0.7)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.fill_between(x_um, norm_residuals, alpha=0.3)
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('Normalized Residuals')
    ax.set_title('Residuals in Normalized Space')
    ax.grid(True, alpha=0.3)
    
    # 9. Parameters
    ax = axes[2, 2]
    ax.axis('off')
    
    text = "Parameters:\n" + "="*30 + "\n"
    text += f"offset0  = {parameters['offset0']:.6e}\n"
    text += f"offset1  = {parameters['offset1']:.6e}\n"
    text += f"amp0     = {parameters['amp0']:.6e}\n"
    text += f"amp1     = {parameters['amp1']:.6e}\n"
    text += f"beta     = {parameters['beta']:.6e}\n"
    text += f"phi0     = {parameters['phi0']:.6e}\n"
    text += "\nQuality:\n" + "="*30 + "\n"
    text += f"RMSE     = {quality['rmse']:.6e}\n"
    text += f"R²       = {quality['r_squared']:.6f}\n"
    text += f"Max err  = {quality['max_error']:.6e}\n"
    text += f"Period   = {quality['period_um']:.4f} μm\n"
    text += "\nModel:\n" + "="*30 + "\n"
    text += "norm = offset(x) + amp(x)×sin()\n"
    text += "SNOM = envelope × norm"
    
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
    print("Testing Ultimate SNOM Fitter")
    
    # Generate complex test data
    np.random.seed(42)
    x = np.linspace(0, 20e-6, 2000)
    
    # Complex envelope
    envelope_true = 0.1 + 4e-6*x + 1e-10*x**2
    
    # Varying offset and amplitude in normalized signal
    offset_true = 0.5 + 3e-5*x
    amp_true = 0.5 + 2e-5*x
    
    # Oscillation
    beta_true = 1e6
    phi0_true = 0.5
    
    normalized_true = offset_true + amp_true * np.sin(2*beta_true*x + phi0_true)
    snom_true = envelope_true * normalized_true
    snom_signal = snom_true + 0.01 * np.random.randn(len(x))
    
    print(f"\nTest data: {len(x)} points")
    print(f"  Varying offset: {offset_true[0]:.4f} to {offset_true[-1]:.4f}")
    print(f"  Varying amplitude: {amp_true[0]:.4f} to {amp_true[-1]:.4f}")
    
    # Fit
    fitted, params, res, qual, comps = fit_snom_ultimate(
        x, snom_signal, trim_edges=0.1
    )
    
    # Plot
    plot_ultimate_fit(x, snom_signal, fitted, res, qual, params, comps,
                     save_path='/mnt/user-data/outputs/ultimate_snom_fit.png')
    
    print("\n" + "="*70)
    print("✓ Test complete!")
    print("="*70)

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
    
    # Test 3: Fit with edge trimming
    print("\n" + "="*70)
    print("TEST 3: Fit with EDGE TRIMMING (trim 10% from each side)")
    print("="*70)
    
    fitted3, params3, res3, qual3, env3, norm3 = fit_snom_full(
        x, snom_signal,
        envelope_degree=2,
        method='robust',
        trim_edges=0.1  # Trim 10% from each edge
    )
    
    print(f"R² = {qual3['r_squared']:.6f}")
    print(f"RMSE = {qual3['rmse']:.6e}")
    print(f"Period = {qual3['period_um']:.4f} μm")
    
    plot_fit(x, snom_signal, fitted3, res3, qual3, params3,
            env3, norm3,
            save_path='/mnt/user-data/outputs/better_snom_trimmed.png')
    
    # Test 4: Hilbert transform method
    print("\n" + "="*70)
    print("TEST 4: Hilbert Transform (automatic envelope!)")
    print("="*70)
    
    fitted4, params4, res4, qual4, env4, norm4 = fit_snom_hilbert(
        x, snom_signal,
        trim_edges=0.1,
        envelope_degree=2
    )
    
    print(f"R² = {qual4['r_squared']:.6f}")
    print(f"RMSE = {qual4['rmse']:.6e}")
    print(f"Period = {qual4['period_um']:.4f} μm")
    
    plot_fit(x, snom_signal, fitted4, res4, qual4, params4,
            env4, norm4,
            save_path='/mnt/user-data/outputs/better_snom_hilbert.png')
    
    print("\n" + "="*70)
    print("COMPARISON:")
    print("="*70)
    print(f"Known envelope:     R² = {qual['r_squared']:.6f}")
    print(f"Full robust fit:    R² = {qual2['r_squared']:.6f}")
    print(f"With edge trim:     R² = {qual3['r_squared']:.6f}")
    print(f"Hilbert transform:  R² = {qual4['r_squared']:.6f}")
    
    print("\n" + "="*70)
    print("✓ Test complete!")
    print("="*70)