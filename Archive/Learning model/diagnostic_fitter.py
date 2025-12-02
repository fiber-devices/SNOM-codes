"""
Diagnostic Script for Fitting Problems
=======================================
Use this to figure out why your fit is failing (R² = -0.0)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import sys
sys.path.append('/mnt/user-data/outputs')


def diagnose_fitting_problem(time, mi_signal, save_path='/mnt/user-data/outputs/diagnostic_report.png'):
    """
    Comprehensive diagnostic to identify fitting issues.
    """
    print("=" * 70)
    print("DIAGNOSTIC REPORT - Why is the fit failing?")
    print("=" * 70)
    
    time = np.array(time)
    mi_signal = np.array(mi_signal)
    
    # Check 1: Basic data validity
    print("\n1. DATA VALIDITY CHECK")
    print("-" * 70)
    
    has_nan = np.any(np.isnan(mi_signal))
    has_inf = np.any(np.isinf(mi_signal))
    all_same = np.all(mi_signal == mi_signal[0])
    
    print(f"  Contains NaN: {has_nan}")
    print(f"  Contains Inf: {has_inf}")
    print(f"  All values identical: {all_same}")
    print(f"  Number of points: {len(mi_signal)}")
    print(f"  Time range: {time[0]:.6f} to {time[-1]:.6f} s")
    print(f"  Duration: {time[-1] - time[0]:.6f} s")
    
    if has_nan or has_inf:
        print("  ❌ PROBLEM: Data contains NaN or Inf values!")
        return
    
    if all_same:
        print("  ❌ PROBLEM: All signal values are identical!")
        return
    
    print("  ✓ Data validity OK")
    
    # Check 2: Signal characteristics
    print("\n2. SIGNAL CHARACTERISTICS")
    print("-" * 70)
    
    mi_mean = np.mean(mi_signal)
    mi_std = np.std(mi_signal)
    mi_min = np.min(mi_signal)
    mi_max = np.max(mi_signal)
    mi_range = mi_max - mi_min
    
    print(f"  Mean: {mi_mean:.6f}")
    print(f"  Std Dev: {mi_std:.6f}")
    print(f"  Min: {mi_min:.6f}")
    print(f"  Max: {mi_max:.6f}")
    print(f"  Range: {mi_range:.6f}")
    print(f"  SNR (range/std): {mi_range/mi_std:.2f}")
    
    if mi_std < 0.001:
        print("  ⚠ WARNING: Very low variation - signal is nearly flat!")
    
    if mi_range < 0.1:
        print("  ⚠ WARNING: Very small range - are units correct?")
    
    # Check 3: Oscillation detection
    print("\n3. OSCILLATION DETECTION")
    print("-" * 70)
    
    from scipy.signal import find_peaks
    
    peaks, _ = find_peaks(mi_signal, distance=len(mi_signal)//20)
    troughs, _ = find_peaks(-mi_signal, distance=len(mi_signal)//20)
    
    print(f"  Peaks detected: {len(peaks)}")
    print(f"  Troughs detected: {len(troughs)}")
    print(f"  Estimated cycles: {(len(peaks) + len(troughs))/2:.1f}")
    
    if len(peaks) < 2 and len(troughs) < 2:
        print("  ❌ PROBLEM: No oscillations detected - signal appears flat or monotonic!")
        print("  → Check if this is actually an interferometer signal")
    else:
        print("  ✓ Oscillations detected")
        
        if len(peaks) > 0:
            avg_peak_spacing = np.mean(np.diff(time[peaks]))
            print(f"  Average peak spacing: {avg_peak_spacing:.6f} s")
            print(f"  Estimated frequency: {1/avg_peak_spacing:.2f} Hz")
    
    # Check 4: Phase extraction
    print("\n4. PHASE EXTRACTION CHECK")
    print("-" * 70)
    
    try:
        mi_centered = mi_signal - np.mean(mi_signal)
        analytic_signal = hilbert(mi_centered)
        phase = np.unwrap(np.angle(analytic_signal))
        
        phase_range = np.ptp(phase)
        phase_cycles = phase_range / (2 * np.pi)
        
        print(f"  Phase range: {phase_range:.2f} rad")
        print(f"  Phase cycles: {phase_cycles:.2f}")
        
        if phase_cycles < 0.5:
            print("  ⚠ WARNING: Less than 1 cycle detected - signal too short?")
        elif phase_cycles > 100:
            print("  ⚠ WARNING: Very many cycles - check wavelength parameter")
        else:
            print("  ✓ Phase extraction OK")
        
        # Check phase derivative (should vary smoothly)
        phase_deriv = np.gradient(phase, time)
        phase_deriv_std = np.std(phase_deriv)
        
        print(f"  Phase rate (mean): {np.mean(phase_deriv):.2f} rad/s")
        print(f"  Phase rate (std): {phase_deriv_std:.2f} rad/s")
        
        if phase_deriv_std / abs(np.mean(phase_deriv)) > 2.0:
            print("  ⚠ WARNING: Highly variable phase rate - non-smooth motion?")
        
    except Exception as e:
        print(f"  ❌ PROBLEM: Phase extraction failed: {e}")
        phase = None
    
    # Check 5: Normalized signal range
    print("\n5. NORMALIZATION CHECK")
    print("-" * 70)
    
    mi_normalized = (mi_signal - mi_min) / (mi_max - mi_min)
    
    print(f"  Normalized min: {mi_normalized.min():.6f}")
    print(f"  Normalized max: {mi_normalized.max():.6f}")
    
    # Check if it looks like interferometer signal (should swing from ~0 to ~2*mean)
    expected_min = 0
    expected_max = 2 * mi_mean
    actual_swing = mi_max - mi_min
    expected_swing = expected_max - expected_min
    
    print(f"  Expected swing (for I=I0[1+V*cos(φ)]): ~{expected_swing:.4f}")
    print(f"  Actual swing: {actual_swing:.4f}")
    print(f"  Ratio: {actual_swing/expected_swing:.2f}")
    
    if actual_swing / expected_swing < 0.5:
        print("  ⚠ WARNING: Swing is smaller than expected - low visibility?")
    
    # Check 6: Time spacing
    print("\n6. TIME SPACING CHECK")
    print("-" * 70)
    
    dt = np.diff(time)
    dt_mean = np.mean(dt)
    dt_std = np.std(dt)
    
    print(f"  Mean time step: {dt_mean:.9f} s")
    print(f"  Std of time step: {dt_std:.9f} s")
    print(f"  Sampling rate: {1/dt_mean:.2f} Hz")
    
    if dt_std / dt_mean > 0.01:
        print("  ⚠ WARNING: Irregular time spacing")
    else:
        print("  ✓ Time spacing OK")
    
    # Create diagnostic plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # 1. Raw signal
    ax = axes[0, 0]
    ax.plot(time, mi_signal, 'b-', linewidth=1)
    if len(peaks) > 0:
        ax.plot(time[peaks], mi_signal[peaks], 'ro', markersize=6, label='Peaks')
    if len(troughs) > 0:
        ax.plot(time[troughs], mi_signal[troughs], 'go', markersize=6, label='Troughs')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('MI Signal')
    ax.set_title('Raw Signal with Peaks/Troughs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Signal histogram
    ax = axes[0, 1]
    ax.hist(mi_signal, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=mi_mean, color='r', linestyle='--', linewidth=2, label='Mean')
    ax.set_xlabel('MI Signal Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Signal Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Phase (if extracted)
    ax = axes[1, 0]
    if phase is not None:
        ax.plot(time, phase, 'g-', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Phase (rad)')
        ax.set_title(f'Extracted Phase ({phase_cycles:.1f} cycles)')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Phase extraction failed', 
               ha='center', va='center', transform=ax.transAxes)
    
    # 4. Phase derivative
    ax = axes[1, 1]
    if phase is not None:
        phase_deriv = np.gradient(phase, time)
        ax.plot(time, phase_deriv, 'm-', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Phase Rate (rad/s)')
        ax.set_title('Phase Derivative (Velocity)')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Phase extraction failed', 
               ha='center', va='center', transform=ax.transAxes)
    
    # 5. FFT
    ax = axes[2, 0]
    from scipy.fft import fft, fftfreq
    n = len(mi_signal)
    fft_vals = fft(mi_signal - mi_mean)
    fft_freq = fftfreq(n, dt_mean)
    pos_mask = fft_freq > 0
    ax.semilogy(fft_freq[pos_mask], np.abs(fft_vals[pos_mask]))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.set_title('Frequency Spectrum')
    ax.grid(True, alpha=0.3)
    
    # 6. Time spacing
    ax = axes[2, 1]
    ax.plot(dt, 'b-', linewidth=1)
    ax.axhline(y=dt_mean, color='r', linestyle='--', linewidth=2, label='Mean')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Time Step (s)')
    ax.set_title('Time Spacing Uniformity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nDiagnostic plot saved: {save_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    
    issues_found = []
    
    if has_nan or has_inf:
        issues_found.append("Data contains invalid values (NaN/Inf)")
    if all_same:
        issues_found.append("All signal values are identical")
    if len(peaks) < 2 and len(troughs) < 2:
        issues_found.append("No oscillations detected")
    if mi_std < 0.001:
        issues_found.append("Signal has very low variation")
    if phase is not None and phase_cycles < 0.5:
        issues_found.append("Less than 1 cycle in signal")
    
    if len(issues_found) == 0:
        print("\n✓ No major issues detected!")
        print("\nIf fit is still failing, try:")
        print("  1. Use the improved_phase_fitter.py instead")
        print("  2. Check wavelength parameter (currently assuming 632.8 nm)")
        print("  3. Ensure training data and test data are similar")
    else:
        print("\n❌ Issues found:")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
        
        print("\nRecommended actions:")
        if "No oscillations detected" in issues_found:
            print("  → Verify this is an interferometer signal")
            print("  → Check if data preprocessing removed oscillations")
        if "very low variation" in issues_found:
            print("  → Check signal units and scaling")
            print("  → Verify sensor is working correctly")
    
    return fig


# Quick test function
if __name__ == "__main__":
    print("\nThis script diagnoses fitting problems.")
    print("\nUsage:")
    print("  from diagnostic_fitter import diagnose_fitting_problem")
    print("  diagnose_fitting_problem(time, mi_signal)")
    print("\nRunning test with synthetic data...")
    
    # Test with good data
    np.random.seed(42)
    time = np.linspace(0, 0.2, 200)
    piezo = 1.5e-6 * (time / 0.2)
    phase = 4 * np.pi * piezo / 632.8e-9
    mi_signal = 1 + 0.9 * np.cos(phase)
    mi_signal += 0.02 * np.random.randn(len(time))
    
    diagnose_fitting_problem(time, mi_signal)
