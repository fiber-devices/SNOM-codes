# 🎯 FINAL SOLUTION - Triangle Wave Scan Workflow

## Your Exact Situation

```python
## Your data preparation
t, mi = load_data("../data/m400_.npy", 2, 0, 50)
```

- ✅ Triangle wave scans: 0.5 s round trip
- ✅ Forward: 0-0.25 s, Backward: 0.25-0.5 s  
- ✅ Want to split into 80 samples (40 forward + 40 backward)
- ✅ Build nonlinear phase model
- ✅ Fit new signals to minimize residuals

---

## 📁 Your Files (Final Set)

### 🌟 **PRIMARY FILES** (Use These!)

| File | Purpose |
|------|---------|
| **your_workflow.py** | Ready for your exact code ⭐ |
| **prepare_training_data.py** | Triangle wave splitting tool |
| **nonlinear_phase_fitter.py** | Phase model & fitting |
| **NONLINEAR_PHASE_GUIDE.md** | Complete documentation |

### 📊 **Example Outputs**
- scan_splitting.png - Shows how data is split
- my_80_samples.png - All training samples
- my_learned_model.png - Nonlinear phase model
- my_fit.png - Fitting results

---

## 🚀 Quick Start (Your Exact Code)

### Step 1: Prepare Data
```python
from prepare_training_data import split_triangle_scans, combine_samples_for_training

# Load your data (your existing function)
t, mi = load_data("../data/m400_.npy", 2, 0, 50)

# Split triangle wave into forward and backward scans
forward_samples, backward_samples = split_triangle_scans(
    t, mi,
    cycle_duration=0.5,  # Adjust if your round trip is different
    plot=True
)

# Combine into 80 training samples
training_samples = combine_samples_for_training(
    forward_samples, 
    backward_samples, 
    use_both=True  # 80 samples (40F + 40B)
)
```

### Step 2: Build Model
```python
from nonlinear_phase_fitter import NonlinearPhaseFitter

fitter = NonlinearPhaseFitter()
fitter.load_training_data(training_samples)
fitter.build_phase_model(model_type='polynomial', degree=7)
```

### Step 3: Fit New Signals
```python
# Load new data and fit
time_new, mi_new = load_new_data()
fit = fitter.fit_new_signal(time_new, mi_new, method='optimize')

# Results
print(f"RMSE: {fit['rmse']}")
print(f"R²: {fit['r_squared']}")
fitted_signal = fit['fitted_signal']
residuals = fit['residuals']
```

---

## 📊 What You Get from Splitting

```
Original Data: 
├─ t, mi (continuous triangle wave)
│
└─> split_triangle_scans()
    │
    ├─ Forward scans: 40 samples
    │  ├─ Sample 0: 0.0 - 0.25 s
    │  ├─ Sample 1: 0.5 - 0.75 s
    │  ├─ Sample 2: 1.0 - 1.25 s
    │  └─ ... (40 total)
    │
    └─ Backward scans: 40 samples
       ├─ Sample 0: 0.25 - 0.5 s
       ├─ Sample 1: 0.75 - 1.0 s
       ├─ Sample 2: 1.25 - 1.5 s
       └─ ... (40 total)

Combined: 80 training samples
```

---

## 🔧 Key Parameters

### Triangle Wave Splitting
```python
split_triangle_scans(
    time,
    mi_signal,
    cycle_duration=0.5,  # <-- ADJUST FOR YOUR SCAN
    plot=True            # Visualize the split
)
```

**Important:** If your round trip time is NOT 0.5 s, change `cycle_duration`!

### Model Building
```python
build_phase_model(
    model_type='polynomial',  # or 'fourier', 'spline', 'ml'
    degree=7                  # 5-9 typical range
)
```

**Recommendation:** Start with polynomial degree 7

### Fitting
```python
fit_new_signal(
    time,
    mi_signal,
    wavelength=632.8e-9,  # Your laser wavelength
    method='optimize'      # or 'direct' (faster)
)
```

**For best residual minimization:** Use `method='optimize'`

---

## 📈 Complete Workflow Diagram

```
┌─────────────────────────────────────────┐
│  YOUR DATA                              │
│  t, mi = load_data("m400_.npy", ...)   │
│  (continuous triangle wave)             │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  SPLIT INTO 80 SAMPLES                  │
│  forward_samples, backward_samples      │
│  = split_triangle_scans(t, mi)         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  COMBINE FOR TRAINING                   │
│  training_samples = combine_...()       │
│  (40 forward + 40 backward = 80)       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  BUILD NONLINEAR PHASE MODEL            │
│  fitter.load_training_data(...)        │
│  fitter.build_phase_model(...)         │
│  (learns phase-signal relationship)     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  FIT NEW SIGNALS                        │
│  fit = fitter.fit_new_signal(...)      │
│  (minimizes residuals)                  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  RESULTS                                │
│  • fitted_signal                        │
│  • residuals (minimized!)              │
│  • phase                                │
│  • RMSE, R²                            │
└─────────────────────────────────────────┘
```

---

## 🎨 Visualization Guide

### 1. scan_splitting.png
**What it shows:**
- Top: Original signal with cycle boundaries marked
- Middle: First 6 forward scans overlaid
- Bottom: First 6 backward scans overlaid

**What to check:**
- Are cycles properly aligned?
- Do forward/backward scans look consistent?

### 2. my_80_samples.png
**What it shows:**
- 3×3 grid of 9 sample scans from your 80

**What to check:**
- Signal quality across samples
- Any outliers or noise

### 3. my_learned_model.png  
**What it shows:**
- Learned nonlinear phase relationship
- Phase derivative (nonlinearity strength)
- Phase curvature
- Reconstruction quality

**What to check:**
- Smooth, continuous relationship?
- Nonlinearity visible in derivative?

### 4. my_fit.png
**What it shows:**
- Original vs fitted signal
- Residuals over time
- Actual vs fitted scatter
- Residual distribution

**What to check:**
- Fitted signal matches original?
- Residuals random (no patterns)?
- Low RMSE value?

---

## ✅ Quality Checks

### Good Splitting
✅ Forward and backward scan counts match  
✅ All scans have similar duration  
✅ No large duration outliers  
✅ Consistent MI signal ranges  

### Good Model
✅ RMSE during training < 20 rad  
✅ Phase model looks smooth  
✅ No discontinuities  

### Good Fit
✅ R² > 0.90 (ideally > 0.95)  
✅ RMSE < 10% of signal range  
✅ Residuals randomly distributed  
✅ No systematic patterns  

---

## 🔧 Troubleshooting

### Issue: Wrong number of samples
**Cause:** Incorrect `cycle_duration` parameter  
**Solution:**
```python
# Calculate from your data
actual_cycle = (t[-1] - t[0]) / expected_number_of_cycles
split_triangle_scans(t, mi, cycle_duration=actual_cycle)
```

### Issue: Scans don't look aligned
**Cause:** Data doesn't start at beginning of cycle  
**Solution:**
```python
# Manually trim to first cycle start
start_idx = find_first_peak(mi)
t_trimmed = t[start_idx:]
mi_trimmed = mi[start_idx:]
```

### Issue: Poor fit quality (low R²)
**Cause:** Model too simple or wrong type  
**Solutions:**
1. Increase degree: `degree=9` instead of `7`
2. Try different model: `model_type='ml'`
3. Use 'optimize' method instead of 'direct'
4. Check if new signal is similar to training data

---

## 💾 Saving & Reusing

### Save After Training (Once)
```python
import pickle

# Save trained model
with open('my_trained_fitter.pkl', 'wb') as f:
    pickle.dump(fitter, f)
```

### Load and Use (Many Times)
```python
import pickle

# Load
with open('my_trained_fitter.pkl', 'rb') as f:
    fitter = pickle.load(f)

# Use immediately on new data
t_new, mi_new = load_data("new_scan.npy", 2, 0, 50)
fit = fitter.fit_new_signal(t_new, mi_new)
```

---

## 🔄 Batch Processing

```python
# Process multiple files
files = ["scan001.npy", "scan002.npy", "scan003.npy", ...]

results = []
for filename in files:
    # Load
    t, mi = load_data(filename, 2, 0, 50)
    
    # Split if triangle wave
    forward, backward = split_triangle_scans(t, mi, cycle_duration=0.5, plot=False)
    
    # Fit all forward scans
    for i, scan in enumerate(forward):
        fit = fitter.fit_new_signal(scan['time'], scan['mi_signal'])
        results.append({
            'file': filename,
            'scan': i,
            'direction': 'forward',
            'rmse': fit['rmse'],
            'r2': fit['r_squared']
        })

# Save results
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('batch_results.csv', index=False)
```

---

## 📖 File Reference

| File | Lines | Purpose |
|------|-------|---------|
| your_workflow.py | ~150 | **START HERE** - Your exact workflow |
| prepare_training_data.py | ~400 | Triangle wave splitting |
| nonlinear_phase_fitter.py | ~500 | Phase model & fitting |
| complete_workflow.py | ~300 | Full example with all steps |

---

## 🎯 Summary

**You have everything you need to:**
1. ✅ Load your triangle wave data: `t, mi = load_data(...)`
2. ✅ Split into 80 samples: `split_triangle_scans()`
3. ✅ Build nonlinear phase model: `build_phase_model()`
4. ✅ Fit new signals: `fit_new_signal()`
5. ✅ Minimize residuals automatically
6. ✅ Save and reuse the model

**Start with: `your_workflow.py`**

Just replace the `load_data()` function with your actual implementation and you're ready to go! 🚀
