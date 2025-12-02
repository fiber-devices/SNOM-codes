"""
Complete Workflow: From Triangle Wave Data to Fitted Signals
=============================================================
1. Load your triangle wave scan data
2. Split into 80 samples (40 forward + 40 backward)
3. Build nonlinear phase model
4. Fit new signals
"""

import numpy as np
import sys
sys.path.append('/mnt/user-data/outputs')

from prepare_training_data import split_triangle_scans, combine_samples_for_training, analyze_scan_consistency
from nonlinear_phase_fitter import NonlinearPhaseFitter

print("=" * 70)
print("COMPLETE WORKFLOW: TRIANGLE WAVE TO NONLINEAR PHASE MODEL")
print("=" * 70)

# =============================================================================
# STEP 1: LOAD YOUR DATA
# =============================================================================

print("\n" + "=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

# YOUR LOADING FUNCTION
def load_data(filename, arg2, arg3, arg4):
    """
    Replace this with your actual load_data function.
    Should return (time, mi_signal) arrays.
    """
    # Example placeholder
    data = np.load(filename)
    # Your processing logic here
    time = data  # Modify based on your data structure
    mi_signal = data  # Modify based on your data structure
    return time, mi_signal


# Load your data
print("\n⚠ REPLACE THIS WITH YOUR ACTUAL DATA LOADING:")
print('   t, mi = load_data("../data/m400_.npy", 2, 0, 50)')

# For demonstration, generate synthetic data
print("\nUsing synthetic data for demonstration...")

# Generate synthetic triangle wave data
n_cycles = 40
cycle_duration = 0.5
sample_rate = 1000
wavelength = 632.8e-9

total_duration = n_cycles * cycle_duration
n_points = int(total_duration * sample_rate)
t = np.linspace(0, total_duration, n_points)

piezo_amplitude = 2e-6
triangle_wave = np.abs(2 * ((t % cycle_duration) / cycle_duration - 0.5)) - 0.5
piezo = piezo_amplitude * triangle_wave
piezo += 0.1 * piezo_amplitude * np.sin(4 * np.pi * piezo / piezo_amplitude)

phase = 4 * np.pi * piezo / wavelength
mi = 1 + 0.9 * np.cos(phase)
mi += 0.02 * np.random.randn(n_points)

print(f"✓ Loaded data:")
print(f"  Duration: {t[-1] - t[0]:.3f} s")
print(f"  Points: {len(t)}")
print(f"  MI range: [{mi.min():.4f}, {mi.max():.4f}]")

# =============================================================================
# STEP 2: SPLIT INTO TRAINING SAMPLES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: SPLITTING INTO FORWARD/BACKWARD SCANS")
print("=" * 70)

# Split the triangle wave into forward and backward scans
forward_samples, backward_samples = split_triangle_scans(
    t, mi,
    cycle_duration=0.5,  # Adjust this to match your scan duration
    plot=True
)

# Analyze consistency
print("\n" + "-" * 70)
analyze_scan_consistency(forward_samples, 'forward')
analyze_scan_consistency(backward_samples, 'backward')

# Combine samples
# Option 1: Use both forward and backward (80 samples)
training_samples = combine_samples_for_training(
    forward_samples, 
    backward_samples, 
    use_both=True  # Set to False to use only 40 forward scans
)

print(f"\n✓ Created {len(training_samples)} training samples")

# =============================================================================
# STEP 3: BUILD NONLINEAR PHASE MODEL
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: BUILDING NONLINEAR PHASE MODEL")
print("=" * 70)

# Create fitter
fitter = NonlinearPhaseFitter()

# Load training data
fitter.load_training_data(training_samples)

# Visualize training samples
fitter.plot_training_data(save_path='/mnt/user-data/outputs/all_training_samples.png')

# Build phase model
print("\nTesting different model types...")
model_results = {}

for model_type in ['polynomial', 'fourier', 'ml']:
    print(f"\n--- Testing {model_type} ---")
    try:
        fitter.build_phase_model(model_type=model_type, degree=7)
        # Store for comparison (RMSE is printed by the function)
        model_results[model_type] = 'success'
    except Exception as e:
        print(f"  Failed: {e}")
        model_results[model_type] = 'failed'

# Use best model (polynomial degree 7 is usually good)
print("\n" + "-" * 70)
print("Building final model with polynomial degree 7...")
fitter.build_phase_model(model_type='polynomial', degree=7)

# Visualize phase model
fitter.plot_phase_model(save_path='/mnt/user-data/outputs/learned_phase_model.png')

print("\n✓ Nonlinear phase model trained successfully!")

# =============================================================================
# STEP 4: FIT A NEW SIGNAL
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: FITTING NEW MI SIGNAL")
print("=" * 70)

# Generate a new test signal (replace with your actual new data)
print("\nGenerating test signal...")
print("⚠ Replace this with your actual new signal to fit!")

time_new = np.linspace(0, 0.25, 250)  # One forward scan
piezo_new = 1.8e-6 * (time_new / 0.25)  # Linear ramp
piezo_new += 0.15 * 1.8e-6 * np.sin(4 * np.pi * piezo_new / 1.8e-6)  # Add nonlinearity
phase_new = 4 * np.pi * piezo_new / wavelength
mi_new = 1 + 0.87 * np.cos(phase_new)
mi_new += 0.025 * np.random.randn(len(time_new))

print(f"  Duration: {time_new[-1]:.3f} s")
print(f"  Points: {len(time_new)}")
print(f"  MI range: [{mi_new.min():.4f}, {mi_new.max():.4f}]")

# Fit using the learned model
print("\nFitting signal to minimize residuals...")
fit_result = fitter.fit_new_signal(
    time_new,
    mi_new,
    wavelength=632.8e-9,
    method='optimize'  # Use 'direct' for faster fitting
)

# Visualize fit
fitter.plot_fit_result(
    fit_result,
    time_new,
    mi_new,
    save_path='/mnt/user-data/outputs/final_fit_result.png'
)

# =============================================================================
# STEP 5: SAVE EVERYTHING
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: SAVING RESULTS")
print("=" * 70)

# Save fit results
import pandas as pd

results_df = pd.DataFrame({
    'time': time_new,
    'mi_original': mi_new,
    'mi_fitted': fit_result['fitted_signal'],
    'residuals': fit_result['residuals'],
    'phase': fit_result['phase']
})
results_df.to_csv('/mnt/user-data/outputs/complete_fit_results.csv', index=False)
print("  ✓ Saved: complete_fit_results.csv")

# Save summary
summary = {
    'n_training_samples': len(training_samples),
    'model_type': 'polynomial',
    'model_degree': 7,
    'fit_rmse': fit_result['rmse'],
    'fit_r_squared': fit_result['r_squared'],
    'I0': fit_result['I0'],
    'visibility': fit_result['visibility']
}
summary_df = pd.DataFrame([summary])
summary_df.to_csv('/mnt/user-data/outputs/complete_summary.csv', index=False)
print("  ✓ Saved: complete_summary.csv")

# Save trained fitter
import pickle
with open('/mnt/user-data/outputs/complete_trained_fitter.pkl', 'wb') as f:
    pickle.dump(fitter, f)
print("  ✓ Saved: complete_trained_fitter.pkl")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("COMPLETE! 🎉")
print("=" * 70)

print(f"""
Training Phase Complete:
  • Loaded triangle wave data: {len(t)} points over {t[-1]:.1f} s
  • Split into {len(training_samples)} samples ({len(forward_samples)} forward + {len(backward_samples)} backward)
  • Built nonlinear phase model: polynomial degree 7
  
Fitting Phase Complete:
  • Fitted new signal: {len(time_new)} points
  • RMSE: {fit_result['rmse']:.6f}
  • R² Score: {fit_result['r_squared']:.6f}
  • Max residual: {np.max(np.abs(fit_result['residuals'])):.6f}

Generated Files:
  1. scan_splitting.png         - Triangle wave split visualization
  2. all_training_samples.png   - All {len(training_samples)} training samples
  3. learned_phase_model.png    - Nonlinear phase relationship
  4. final_fit_result.png       - Fitting results (4 panels)
  5. complete_fit_results.csv   - Numerical results
  6. complete_summary.csv       - Summary statistics
  7. complete_trained_fitter.pkl - Reusable trained model

Quality Assessment:
""")

if fit_result['r_squared'] > 0.95:
    print("  ✅ Excellent fit (R² > 0.95)")
elif fit_result['r_squared'] > 0.90:
    print("  ✅ Very good fit (R² > 0.90)")
elif fit_result['r_squared'] > 0.80:
    print("  👍 Good fit (R² > 0.80)")
else:
    print("  ⚠  Fair fit - consider adjusting parameters")

signal_range = np.ptp(mi_new)
relative_rmse = fit_result['rmse'] / signal_range
if relative_rmse < 0.05:
    print(f"  ✅ Low residuals ({relative_rmse*100:.1f}% of signal range)")
elif relative_rmse < 0.10:
    print(f"  👍 Moderate residuals ({relative_rmse*100:.1f}% of signal range)")
else:
    print(f"  ⚠  High residuals ({relative_rmse*100:.1f}% of signal range)")

print("""
Next Steps:
  1. Review all visualization files
  2. If satisfied with fit quality, use the trained model for new signals:
  
     import pickle
     with open('complete_trained_fitter.pkl', 'rb') as f:
         fitter = pickle.load(f)
     
     # Fit new signal
     fit = fitter.fit_new_signal(time, mi_signal, method='optimize')
  
  3. For batch processing, loop over multiple signals
  4. Save all results for analysis
""")

print("\n" + "=" * 70)
