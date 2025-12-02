"""
YOUR SPECIFIC WORKFLOW
======================
Adapted for your exact code structure with triangle wave scans.
"""

import numpy as np
import sys
sys.path.append('/mnt/user-data/outputs')

from prepare_training_data import split_triangle_scans, combine_samples_for_training
from nonlinear_phase_fitter import NonlinearPhaseFitter

# =============================================================================
# YOUR DATA LOADING FUNCTION
# =============================================================================

def load_data(filename, arg2, arg3, arg4):
    """
    YOUR ACTUAL LOAD_DATA FUNCTION
    Replace this with your implementation!
    """
    # Your code here
    data = np.load(filename)
    # Process and return time and mi_signal
    return time, mi_signal


# =============================================================================
# STEP 1: LOAD YOUR DATA
# =============================================================================

print("Step 1: Loading data...")

## data preparation - YOUR EXACT CODE
t, mi = load_data("../data/m400_.npy", 2, 0, 50)

print(f"  Loaded: {len(t)} points")
print(f"  Duration: {t[-1] - t[0]:.3f} s")
print(f"  MI range: [{mi.min():.4f}, {mi.max():.4f}]")

# =============================================================================
# STEP 2: SPLIT TRIANGLE WAVE INTO 80 SAMPLES
# =============================================================================

print("\nStep 2: Splitting triangle wave...")

# Split into forward and backward scans
# Adjust cycle_duration if your round trip is not 0.5 s
forward_samples, backward_samples = split_triangle_scans(
    t, mi,
    cycle_duration=0.5,  # <-- ADJUST THIS if your round trip time is different
    plot=True
)

print(f"  Forward scans: {len(forward_samples)}")
print(f"  Backward scans: {len(backward_samples)}")

# Combine into training set
# Option 1: Use both forward and backward (80 samples total)
training_samples = combine_samples_for_training(
    forward_samples, 
    backward_samples, 
    use_both=True  # Set False to use only 40 forward scans
)

# Option 2: Use only forward scans (40 samples)
# training_samples = combine_samples_for_training(
#     forward_samples, 
#     backward_samples, 
#     use_both=False
# )

print(f"  Training samples: {len(training_samples)}")

# =============================================================================
# STEP 3: BUILD NONLINEAR PHASE MODEL
# =============================================================================

print("\nStep 3: Building phase model...")

fitter = NonlinearPhaseFitter()
fitter.load_training_data(training_samples)

# Visualize your 80 (or 40) training samples
fitter.plot_training_data(save_path='/mnt/user-data/outputs/my_80_samples.png')

# Build the model
# Try polynomial degree 7 first (good balance)
fitter.build_phase_model(model_type='polynomial', degree=7)

# Visualize the learned model
fitter.plot_phase_model(save_path='/mnt/user-data/outputs/my_learned_model.png')

print("  ✓ Model trained!")

# =============================================================================
# STEP 4: FIT NEW MI SIGNAL
# =============================================================================

print("\nStep 4: Fitting new signal...")

# Load your new signal to fit
# Method 1: If it's another triangle wave scan
# time_new, mi_new = load_data("../data/new_scan.npy", 2, 0, 50)
# # Extract just one forward scan
# forward_new, backward_new = split_triangle_scans(time_new, mi_new, cycle_duration=0.5, plot=False)
# time_to_fit = forward_new[0]['time']
# mi_to_fit = forward_new[0]['mi_signal']

# Method 2: If it's a single scan already
# time_to_fit, mi_to_fit = load_single_scan("../data/single_scan.npy")

# For demonstration (REPLACE THIS):
time_to_fit = training_samples[0]['time']
mi_to_fit = training_samples[0]['mi_signal']

# Fit the signal
fit_result = fitter.fit_new_signal(
    time_to_fit,
    mi_to_fit,
    wavelength=632.8e-9,  # Adjust to your laser wavelength
    method='optimize'      # or 'direct' for faster fitting
)

# Visualize fit
fitter.plot_fit_result(
    fit_result,
    time_to_fit,
    mi_to_fit,
    save_path='/mnt/user-data/outputs/my_fit.png'
)

print(f"  RMSE: {fit_result['rmse']:.6f}")
print(f"  R²: {fit_result['r_squared']:.6f}")

# =============================================================================
# STEP 5: SAVE FOR REUSE
# =============================================================================

print("\nStep 5: Saving model...")

import pickle
with open('/mnt/user-data/outputs/my_fitter.pkl', 'wb') as f:
    pickle.dump(fitter, f)

print("  ✓ Saved: my_fitter.pkl")

# =============================================================================
# LATER: REUSE THE MODEL
# =============================================================================

"""
# In a new session or script:

import pickle
from prepare_training_data import split_triangle_scans

# Load the trained model
with open('my_fitter.pkl', 'rb') as f:
    fitter = pickle.load(f)

# Load new data
t_new, mi_new = load_data("../data/another_scan.npy", 2, 0, 50)

# Split if it's a triangle wave
forward, backward = split_triangle_scans(t_new, mi_new, cycle_duration=0.5, plot=False)

# Fit each scan
for i, scan in enumerate(forward):
    fit = fitter.fit_new_signal(scan['time'], scan['mi_signal'], method='optimize')
    print(f"Scan {i}: RMSE={fit['rmse']:.6f}, R²={fit['r_squared']:.6f}")
    
    # Access results
    fitted_signal = fit['fitted_signal']
    residuals = fit['residuals']
    phase = fit['phase']
"""

print("\n" + "=" * 70)
print("DONE! Check the output files:")
print("  • scan_splitting.png - How your data was split")
print("  • my_80_samples.png - All training samples")
print("  • my_learned_model.png - The nonlinear phase model")
print("  • my_fit.png - Fitting results")
print("  • my_fitter.pkl - Trained model (reusable)")
print("=" * 70)
