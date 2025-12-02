"""
Data Preparation for Triangle Wave Scans
=========================================
Splits continuous triangle wave scan data into separate forward/backward samples
for training the nonlinear phase model.

Triangle wave structure:
- One complete cycle = 0.5 s (round trip)
- Forward scan: 0 to 0.25 s
- Backward scan: 0.25 to 0.5 s
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(filename, arg2, arg3, arg4):
    """
    Your existing data loading function.
    Replace this with your actual implementation.
    """
    # Placeholder - replace with your actual load_data function
    data = np.load(filename)
    # Your processing logic here based on arguments
    return data  # Should return (time, mi_signal)


def split_triangle_scans(time, mi_signal, cycle_duration=0.5, plot=True, 
                        center_fraction=1.0):
    """
    Split continuous triangle wave scan into separate forward/backward samples.
    
    Parameters:
    -----------
    time : array
        Time array
    mi_signal : array
        MI signal array
    cycle_duration : float
        Duration of one complete triangle cycle (forward + backward) in seconds
    plot : bool
        Whether to create visualization
    center_fraction : float
        Fraction of signal to extract from center (0.0 to 1.0)
        Example: 0.8 extracts central 80%, discarding 10% from each end
    
    Returns:
    --------
    forward_samples : list of dict
        List of forward scan samples, each with {'time': array, 'mi_signal': array}
    backward_samples : list of dict
        List of backward scan samples, each with {'time': array, 'mi_signal': array}
    """
    print("=" * 70)
    print("SPLITTING TRIANGLE WAVE SCANS")
    print("=" * 70)
    
    time = np.array(time)
    mi_signal = np.array(mi_signal)
    
    # Validate center_fraction
    if not 0.0 < center_fraction <= 1.0:
        raise ValueError(f"center_fraction must be between 0 and 1, got {center_fraction}")
    
    # Calculate parameters
    half_cycle = cycle_duration / 2  # Duration of forward or backward scan
    total_duration = time[-1] - time[0]
    n_complete_cycles = int(total_duration / cycle_duration)
    
    print(f"\nData info:")
    print(f"  Total duration: {total_duration:.3f} s")
    print(f"  Cycle duration: {cycle_duration:.3f} s")
    print(f"  Half cycle (forward/backward): {half_cycle:.3f} s")
    print(f"  Complete cycles detected: {n_complete_cycles}")
    print(f"  Total samples: {n_complete_cycles * 2} ({n_complete_cycles} forward + {n_complete_cycles} backward)")
    
    if center_fraction < 1.0:
        discard_fraction = (1.0 - center_fraction) / 2
        print(f"\n⚠ Center extraction enabled:")
        print(f"  Extracting: {center_fraction*100:.1f}% from center")
        print(f"  Discarding: {discard_fraction*100:.1f}% from each end")
        print(f"  Each scan duration: {half_cycle:.3f} s → {half_cycle*center_fraction:.3f} s")
    
    forward_samples = []
    backward_samples = []
    
    # Calculate center extraction parameters
    discard_fraction = (1.0 - center_fraction) / 2
    
    # Split into cycles
    for cycle_idx in range(n_complete_cycles):
        # Time boundaries for this cycle
        cycle_start = time[0] + cycle_idx * cycle_duration
        cycle_mid = cycle_start + half_cycle
        cycle_end = cycle_start + cycle_duration
        
        # Find indices for forward scan (first half)
        forward_mask = (time >= cycle_start) & (time < cycle_mid)
        forward_time = time[forward_mask]
        forward_mi = mi_signal[forward_mask]
        
        # Find indices for backward scan (second half)
        backward_mask = (time >= cycle_mid) & (time < cycle_end)
        backward_time = time[backward_mask]
        backward_mi = mi_signal[backward_mask]
        
        # Extract center portion if requested
        if center_fraction < 1.0 and len(forward_time) > 0:
            n_forward = len(forward_time)
            start_idx = int(n_forward * discard_fraction)
            end_idx = int(n_forward * (1.0 - discard_fraction))
            forward_time = forward_time[start_idx:end_idx]
            forward_mi = forward_mi[start_idx:end_idx]
        
        if center_fraction < 1.0 and len(backward_time) > 0:
            n_backward = len(backward_time)
            start_idx = int(n_backward * discard_fraction)
            end_idx = int(n_backward * (1.0 - discard_fraction))
            backward_time = backward_time[start_idx:end_idx]
            backward_mi = backward_mi[start_idx:end_idx]
        
        # Store samples with normalized time starting from 0
        if len(forward_time) > 0:
            forward_samples.append({
                'time': forward_time - forward_time[0],  # Start from 0
                'mi_signal': forward_mi,
                'cycle_number': cycle_idx,
                'absolute_time_start': cycle_start if center_fraction == 1.0 else cycle_start + half_cycle * discard_fraction
            })
        
        if len(backward_time) > 0:
            backward_samples.append({
                'time': backward_time - backward_time[0],  # Start from 0
                'mi_signal': backward_mi,
                'cycle_number': cycle_idx,
                'absolute_time_start': cycle_mid if center_fraction == 1.0 else cycle_mid + half_cycle * discard_fraction
            })
    
    print(f"\nExtracted samples:")
    print(f"  Forward scans: {len(forward_samples)}")
    print(f"  Backward scans: {len(backward_samples)}")
    print(f"  Total: {len(forward_samples) + len(backward_samples)}")
    
    # Sample statistics
    if len(forward_samples) > 0:
        avg_forward_points = np.mean([len(s['mi_signal']) for s in forward_samples])
        print(f"\nForward scan statistics:")
        print(f"  Average points per scan: {avg_forward_points:.1f}")
        print(f"  First scan duration: {forward_samples[0]['time'][-1]:.6f} s")
        print(f"  First scan MI range: [{forward_samples[0]['mi_signal'].min():.4f}, "
              f"{forward_samples[0]['mi_signal'].max():.4f}]")
    
    if len(backward_samples) > 0:
        avg_backward_points = np.mean([len(s['mi_signal']) for s in backward_samples])
        print(f"\nBackward scan statistics:")
        print(f"  Average points per scan: {avg_backward_points:.1f}")
        print(f"  First scan duration: {backward_samples[0]['time'][-1]:.6f} s")
        print(f"  First scan MI range: [{backward_samples[0]['mi_signal'].min():.4f}, "
              f"{backward_samples[0]['mi_signal'].max():.4f}]")
    
    # Visualization
    if plot:
        plot_scan_splitting(time, mi_signal, forward_samples, backward_samples, 
                          cycle_duration)
    
    return forward_samples, backward_samples


def plot_scan_splitting(time, mi_signal, forward_samples, backward_samples, 
                       cycle_duration, save_path='./DATA/scan_splitting.png'):
    """Create visualization of scan splitting."""
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # 1. Full signal with cycle boundaries
    ax = axes[0]
    ax.plot(time, mi_signal, 'b-', linewidth=1, alpha=0.7, label='Original signal')
    
    # Mark cycle boundaries
    n_cycles = len(forward_samples)
    for i in range(n_cycles + 1):
        t_boundary = time[0] + i * cycle_duration
        ax.axvline(x=t_boundary, color='red', linestyle='--', alpha=0.5, linewidth=1)
        if i < n_cycles:
            # Mark mid-cycle (forward/backward transition)
            t_mid = t_boundary + cycle_duration / 2
            ax.axvline(x=t_mid, color='green', linestyle=':', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('MI Signal', fontsize=11)
    ax.set_title('Original Signal with Cycle Boundaries', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text annotations for first few cycles
    for i in range(min(3, n_cycles)):
        t_start = time[0] + i * cycle_duration
        t_mid = t_start + cycle_duration / 2
        y_pos = ax.get_ylim()[1] * 0.95
        ax.text(t_start + cycle_duration/4, y_pos, f'F{i+1}', 
               ha='center', va='top', fontsize=9, color='blue', fontweight='bold')
        ax.text(t_mid + cycle_duration/4, y_pos, f'B{i+1}', 
               ha='center', va='top', fontsize=9, color='orange', fontweight='bold')
    
    # 2. First few forward scans
    ax = axes[1]
    n_show = min(6, len(forward_samples))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_show))
    
    for i in range(n_show):
        sample = forward_samples[i]
        ax.plot(sample['time'], sample['mi_signal'], 
               color=colors[i], linewidth=1.5, alpha=0.8, label=f'Forward {i+1}')
    
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('MI Signal', fontsize=11)
    ax.set_title(f'First {n_show} Forward Scans (Overlaid)', fontsize=12, fontweight='bold')
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 3. First few backward scans
    ax = axes[2]
    n_show = min(6, len(backward_samples))
    colors = plt.cm.Oranges(np.linspace(0.4, 0.9, n_show))
    
    for i in range(n_show):
        sample = backward_samples[i]
        ax.plot(sample['time'], sample['mi_signal'], 
               color=colors[i], linewidth=1.5, alpha=0.8, label=f'Backward {i+1}')
    
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('MI Signal', fontsize=11)
    ax.set_title(f'First {n_show} Backward Scans (Overlaid)', fontsize=12, fontweight='bold')
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Ensure the directory for the save path exists, create it if it doesn't
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: {save_path}")
    
    return fig


def combine_samples_for_training(forward_samples, backward_samples, use_both=True):
    """
    Combine forward and backward samples for training.
    
    Parameters:
    -----------
    forward_samples : list
        List of forward scan samples
    backward_samples : list
        List of backward scan samples
    use_both : bool
        If True, use both forward and backward (80 samples)
        If False, use only forward (40 samples)
    
    Returns:
    --------
    training_samples : list
        Combined samples for training (40 or 80 samples)
    """
    if use_both:
        # Combine forward and backward scans
        training_samples = []
        
        # Interleave forward and backward for balanced training
        for i in range(min(len(forward_samples), len(backward_samples))):
            training_samples.append({
                'time': forward_samples[i]['time'],
                'mi_signal': forward_samples[i]['mi_signal']
            })
            training_samples.append({
                'time': backward_samples[i]['time'],
                'mi_signal': backward_samples[i]['mi_signal']
            })
        
        print(f"\nCombined {len(training_samples)} samples (forward + backward)")
    else:
        # Use only forward scans
        training_samples = [{
            'time': s['time'],
            'mi_signal': s['mi_signal']
        } for s in forward_samples]
        
        print(f"\nUsing {len(training_samples)} forward samples only")
    
    return training_samples


def analyze_scan_consistency(samples, sample_type='forward'):
    """
    Analyze consistency across scans to detect issues.
    
    Parameters:
    -----------
    samples : list
        List of scan samples
    sample_type : str
        'forward' or 'backward'
    """
    print(f"\n{sample_type.upper()} SCAN CONSISTENCY ANALYSIS")
    print("-" * 70)
    
    # Extract statistics from each sample
    durations = [s['time'][-1] for s in samples]
    n_points = [len(s['mi_signal']) for s in samples]
    mi_means = [np.mean(s['mi_signal']) for s in samples]
    mi_stds = [np.std(s['mi_signal']) for s in samples]
    mi_ranges = [np.ptp(s['mi_signal']) for s in samples]
    
    print(f"Duration statistics:")
    print(f"  Mean: {np.mean(durations):.6f} s")
    print(f"  Std:  {np.std(durations):.6f} s")
    print(f"  Range: [{np.min(durations):.6f}, {np.max(durations):.6f}] s")
    
    print(f"\nPoints per scan:")
    print(f"  Mean: {np.mean(n_points):.1f}")
    print(f"  Std:  {np.std(n_points):.1f}")
    print(f"  Range: [{np.min(n_points)}, {np.max(n_points)}]")
    
    print(f"\nMI signal mean:")
    print(f"  Mean: {np.mean(mi_means):.6f}")
    print(f"  Std:  {np.std(mi_means):.6f}")
    
    print(f"\nMI signal range:")
    print(f"  Mean: {np.mean(mi_ranges):.6f}")
    print(f"  Std:  {np.std(mi_ranges):.6f}")
    
    # Check for outliers
    duration_outliers = np.abs(durations - np.mean(durations)) > 3 * np.std(durations)
    if np.any(duration_outliers):
        outlier_indices = np.where(duration_outliers)[0]
        print(f"\n⚠ Warning: Duration outliers detected at indices: {outlier_indices}")
    
    range_outliers = np.abs(mi_ranges - np.mean(mi_ranges)) > 3 * np.std(mi_ranges)
    if np.any(range_outliers):
        outlier_indices = np.where(range_outliers)[0]
        print(f"⚠ Warning: MI range outliers detected at indices: {outlier_indices}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EXAMPLE: SPLITTING TRIANGLE WAVE SCANS")
    print("=" * 70)
    
    # Generate synthetic triangle wave data for demonstration
    print("\nGenerating synthetic triangle wave data...")
    
    # Parameters
    n_cycles = 40
    cycle_duration = 0.5  # seconds
    sample_rate = 1000  # Hz
    wavelength = 632.8e-9
    
    # Time array
    total_duration = n_cycles * cycle_duration
    n_points = int(total_duration * sample_rate)
    time = np.linspace(0, total_duration, n_points)
    
    # Generate triangle wave piezo movement
    piezo_amplitude = 2e-6  # 2 micrometers
    triangle_wave = np.abs(2 * ((time % cycle_duration) / cycle_duration - 0.5)) - 0.5
    piezo = piezo_amplitude * triangle_wave
    
    # Add some nonlinearity
    piezo += 0.1 * piezo_amplitude * np.sin(4 * np.pi * piezo / piezo_amplitude)
    
    # Generate MI signal
    phase = 4 * np.pi * piezo / wavelength
    mi_signal = 1 + 0.9 * np.cos(phase)
    mi_signal += 0.02 * np.random.randn(n_points)
    
    print(f"  Generated {n_points} points over {total_duration:.1f} s")
    print(f"  {n_cycles} complete cycles")
    
    # Split into forward and backward scans
    forward_samples, backward_samples = split_triangle_scans(
        time, mi_signal, 
        cycle_duration=cycle_duration,
        center_fraction=0.8,  # Extract only central 80% from each scan
        plot=True
    )
    
    # Analyze consistency
    analyze_scan_consistency(forward_samples, 'forward')
    analyze_scan_consistency(backward_samples, 'backward')
    
    # Combine for training (40 forward + 40 backward = 80 samples)
    training_samples_80 = combine_samples_for_training(
        forward_samples, 
        backward_samples, 
        use_both=True
    )
    
    # Or use only forward scans (40 samples)
    training_samples_40 = combine_samples_for_training(
        forward_samples, 
        backward_samples, 
        use_both=False
    )
    
    print("\n" + "=" * 70)
    print("READY FOR TRAINING")
    print("=" * 70)
    print(f"\nYou now have:")
    print(f"  - {len(training_samples_80)} samples (forward + backward)")
    print(f"  - {len(training_samples_40)} samples (forward only)")
    print(f"\nUse these with NonlinearPhaseFitter:")
    print("""
from nonlinear_phase_fitter import NonlinearPhaseFitter

fitter = NonlinearPhaseFitter()
fitter.load_training_data(training_samples_80)  # or training_samples_40
fitter.build_phase_model(model_type='polynomial', degree=7)
    """)
