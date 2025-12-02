import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# 1. PZT + Michelson interferometer simulation
# =========================================================

def simulate_pzt_nonlinear(v, noise_level=1e-9):
    """
    Simulated 'true' PZT displacement model (unknown to ML).

    v : array-like [V]  (applied voltage)
    Returns x_true [m]

    Model: x = a1 v + a3 v^3 + a5 v^5 + small noise
    This is a static nonlinearity, but you can add
    memory/creep later if needed.
    """
    v = np.asarray(v)

    # Coefficients: tune these to your actuator scale
    a1 = 100e-9    # 100 nm per volt (linear gain)
    a3 = -10e-9    # small cubic term
    a5 = 1e-9      # tiny 5th-order term

    x = a1 * v + a3 * (v**3) + a5 * (v**5)

    # Add small Gaussian noise to mimic imperfections
    x += noise_level * np.random.randn(*x.shape)
    return x


def simulate_michelson_signal(x, wavelength=1.55e-6,
                              I0=1.0, visibility=0.9,
                              phi0=0.0, phase_noise_std=0.05):
    """
    Simulate a Michelson interferometer intensity signal.

    x : displacement [m]
    wavelength : laser wavelength [m]
    I0 : average intensity (normalized)
    visibility : fringe contrast (0..1)
    phi0 : static phase offset
    phase_noise_std : std dev of random phase noise [rad]

    Returns I(t)
    """
    x = np.asarray(x)
    k = 2 * np.pi / wavelength
    phi = phi0 + 2 * 2 * k * x   # 4π/λ * x

    # Add random phase noise
    phi_noisy = phi + np.random.randn(*phi.shape) * phase_noise_std

    I = I0 * (1.0 + visibility * np.cos(phi_noisy))
    return I


def generate_dataset(num_waveforms=200,  # Increased for better training
                     points_per_waveform=1000,
                     v_min=0.0, v_max=10.0):
    """
    Generate a dataset of:
      t, v(t), x_true(t), I_MI(t)

    with multiple random waveforms (ramps / triangles / sines etc).
    """
    all_v = []
    all_x = []
    all_I = []

    for _ in range(num_waveforms):
        # Random waveform type: ramp, triangle, sine, or sawtooth
        kind = np.random.choice(["ramp", "triangle", "sine", "sawtooth"])

        # Time axis (normalized 0..1)
        t = np.linspace(0, 1, points_per_waveform)

        if kind == "ramp":
            # Simple monotonic ramp up
            v = v_min + (v_max - v_min) * t
        elif kind == "triangle":
            # Triangle: up then down
            v = v_min + (v_max - v_min) * (2 * np.abs(t - 0.5))
        elif kind == "sine":
            # Sinusoidal waveform
            freq = np.random.uniform(0.5, 3.0)
            v = v_min + (v_max - v_min) * (0.5 + 0.5 * np.sin(2 * np.pi * freq * t))
        else:  # sawtooth
            # Sawtooth waveform
            v = v_min + (v_max - v_min) * (t * 2 % 1.0)

        # Add some random offset and scaling to diversify
        scale = np.random.uniform(0.5, 1.0)
        offset = np.random.uniform(0.0, 0.3 * v_max)
        v = offset + scale * v
        v = np.clip(v, v_min, v_max)

        x_true = simulate_pzt_nonlinear(v, noise_level=1e-10)  # Reduced noise
        I_MI = simulate_michelson_signal(x_true)

        all_v.append(v)
        all_x.append(x_true)
        all_I.append(I_MI)

    v_all = np.concatenate(all_v)[:, None]   # shape (N, 1)
    x_all = np.concatenate(all_x)[:, None]   # shape (N, 1)
    I_all = np.concatenate(all_I)[:, None]   # shape (N, 1)

    return v_all, x_all, I_all


# =========================================================
# 2. Train forward model: v -> x
#    (learn PZT nonlinearity)
# =========================================================

def create_enhanced_features(v):
    """
    Create enhanced feature set including polynomial terms and transformations.
    """
    v = v.flatten()
    features = np.column_stack([
        v,                          # v
        v**2,                      # v²
        v**3,                      # v³
        v**4,                      # v⁴
        v**5,                      # v⁵
        np.sqrt(np.abs(v)),        # √|v|
        np.log1p(np.abs(v)),       # log(1+|v|)
        np.sin(v * np.pi / 10),    # sin(πv/10)
        np.cos(v * np.pi / 10),    # cos(πv/10)
    ])
    return features

def train_forward_model(v_all, x_all):
    """
    Train an enhanced neural network that maps v -> x with R² > 0.999.

    v_all : array (N, 1)
    x_all : array (N, 1)

    Returns trained model and scaler.
    """
    # Create enhanced features
    X_features = create_enhanced_features(v_all)
    y = x_all.flatten()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42
    )
    
    # Scale features and targets
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    
    # Create ensemble of models for better performance
    # Model 1: Deep network with ReLU
    model1 = MLPRegressor(
        hidden_layer_sizes=(256, 512, 256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-5,  # L2 regularization
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=5000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        tol=1e-8,
        random_state=42,
        warm_start=False
    )
    
    # Model 2: Deep network with tanh
    model2 = MLPRegressor(
        hidden_layer_sizes=(128, 256, 128, 64),
        activation="tanh",
        solver="adam",
        alpha=1e-5,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=5000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        tol=1e-8,
        random_state=123,
        warm_start=False
    )
    
    # Model 3: Wider network
    model3 = MLPRegressor(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-6,
        learning_rate="adaptive",
        learning_rate_init=0.0005,
        max_iter=5000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        tol=1e-8,
        random_state=456,
        warm_start=False
    )
    
    # Train all models
    print("Training forward model ensemble...")
    model1.fit(X_train_scaled, y_train_scaled)
    model2.fit(X_train_scaled, y_train_scaled)
    model3.fit(X_train_scaled, y_train_scaled)
    
    # Evaluate individual models
    y_pred1 = scaler_y.inverse_transform(model1.predict(X_test_scaled).reshape(-1, 1)).ravel()
    y_pred2 = scaler_y.inverse_transform(model2.predict(X_test_scaled).reshape(-1, 1)).ravel()
    y_pred3 = scaler_y.inverse_transform(model3.predict(X_test_scaled).reshape(-1, 1)).ravel()
    
    r2_1 = r2_score(y_test, y_pred1)
    r2_2 = r2_score(y_test, y_pred2)
    r2_3 = r2_score(y_test, y_pred3)
    
    print(f"  Model 1 R²: {r2_1:.6f}")
    print(f"  Model 2 R²: {r2_2:.6f}")
    print(f"  Model 3 R²: {r2_3:.6f}")
    
    # Use ensemble prediction (average)
    y_pred_ensemble = (y_pred1 + y_pred2 + y_pred3) / 3.0
    r2_ensemble = r2_score(y_test, y_pred_ensemble)
    print(f"[Forward model] Ensemble R² on test: {r2_ensemble:.6f}")
    
    # Store all models and scalers
    model_dict = {
        'models': [model1, model2, model3],
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'r2': r2_ensemble
    }
    
    return model_dict


# =========================================================
# 3. Train inverse model: x -> v
#    (pre-distortion for linear motion)
# =========================================================

def create_inverse_features(x):
    """
    Create enhanced feature set for inverse model.
    """
    x = x.flatten()
    features = np.column_stack([
        x,                          # x
        x**2,                      # x²
        x**3,                      # x³
        np.sqrt(np.abs(x)),        # √|x|
        np.log1p(np.abs(x)),       # log(1+|x|)
        np.sin(x * 1e6 * np.pi),   # sin(πx scaled)
        np.cos(x * 1e6 * np.pi),   # cos(πx scaled)
    ])
    return features

def train_inverse_model(v_all, x_all):
    """
    Train an enhanced neural network that maps x -> v with R² > 0.999.

    This learns an approximate inverse of the PZT nonlinearity.
    """
    # Create enhanced features
    X_features = create_inverse_features(x_all)
    y = v_all.flatten()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42
    )
    
    # Scale features and targets
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    
    # Create ensemble of models
    # Model 1: Deep network with ReLU
    model1 = MLPRegressor(
        hidden_layer_sizes=(256, 512, 256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-5,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=5000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        tol=1e-8,
        random_state=42,
        warm_start=False
    )
    
    # Model 2: Deep network with tanh
    model2 = MLPRegressor(
        hidden_layer_sizes=(128, 256, 128, 64),
        activation="tanh",
        solver="adam",
        alpha=1e-5,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=5000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        tol=1e-8,
        random_state=123,
        warm_start=False
    )
    
    # Model 3: Wider network
    model3 = MLPRegressor(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-6,
        learning_rate="adaptive",
        learning_rate_init=0.0005,
        max_iter=5000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        tol=1e-8,
        random_state=456,
        warm_start=False
    )
    
    # Train all models
    print("Training inverse model ensemble...")
    model1.fit(X_train_scaled, y_train_scaled)
    model2.fit(X_train_scaled, y_train_scaled)
    model3.fit(X_train_scaled, y_train_scaled)
    
    # Evaluate individual models
    y_pred1 = scaler_y.inverse_transform(model1.predict(X_test_scaled).reshape(-1, 1)).ravel()
    y_pred2 = scaler_y.inverse_transform(model2.predict(X_test_scaled).reshape(-1, 1)).ravel()
    y_pred3 = scaler_y.inverse_transform(model3.predict(X_test_scaled).reshape(-1, 1)).ravel()
    
    r2_1 = r2_score(y_test, y_pred1)
    r2_2 = r2_score(y_test, y_pred2)
    r2_3 = r2_score(y_test, y_pred3)
    
    print(f"  Model 1 R²: {r2_1:.6f}")
    print(f"  Model 2 R²: {r2_2:.6f}")
    print(f"  Model 3 R²: {r2_3:.6f}")
    
    # Use ensemble prediction (average)
    y_pred_ensemble = (y_pred1 + y_pred2 + y_pred3) / 3.0
    r2_ensemble = r2_score(y_test, y_pred_ensemble)
    print(f"[Inverse model] Ensemble R² on test: {r2_ensemble:.6f}")
    
    # Store all models and scalers
    model_dict = {
        'models': [model1, model2, model3],
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'r2': r2_ensemble
    }
    
    return model_dict


# =========================================================
# 4. Use the inverse model to linearize motion
# =========================================================

def predict_forward(model_dict, v):
    """Predict displacement using ensemble forward model."""
    X_features = create_enhanced_features(v)
    X_scaled = model_dict['scaler_X'].transform(X_features)
    
    predictions = []
    for model in model_dict['models']:
        pred_scaled = model.predict(X_scaled)
        predictions.append(pred_scaled)
    
    pred_ensemble_scaled = np.mean(predictions, axis=0)
    pred = model_dict['scaler_y'].inverse_transform(pred_ensemble_scaled.reshape(-1, 1)).ravel()
    return pred

def predict_inverse(model_dict, x):
    """Predict voltage using ensemble inverse model."""
    X_features = create_inverse_features(x)
    X_scaled = model_dict['scaler_X'].transform(X_features)
    
    predictions = []
    for model in model_dict['models']:
        pred_scaled = model.predict(X_scaled)
        predictions.append(pred_scaled)
    
    pred_ensemble_scaled = np.mean(predictions, axis=0)
    pred = model_dict['scaler_y'].inverse_transform(pred_ensemble_scaled.reshape(-1, 1)).ravel()
    return pred

def test_linearization(forward_model, inverse_model,
                       v_min=0.0, v_max=10.0, num_points=1000):
    """
    Generate a desired *linear* displacement trajectory x_desired(t),
    then use the inverse model to compute the voltage command v_cmd(t),
    and simulate the true PZT response x_true(t).

    Compare:
      - x_desired (ideal)
      - x_true_without_comp (using linear ramp of voltage)
      - x_true_with_comp   (using pre-distorted v from inverse model)
    """
    # Desired linear displacement trajectory [m]
    # We'll choose the range based on the nominal linear gain a1=100 nm/V.
    t = np.linspace(0, 1, num_points)
    x_min = simulate_pzt_nonlinear(np.array([v_min]))[0]
    x_max = simulate_pzt_nonlinear(np.array([v_max]))[0]
    x_desired = x_min + (x_max - x_min) * t

    # Case 1: naive linear ramp of voltage
    v_naive = v_min + (v_max - v_min) * t
    x_naive = simulate_pzt_nonlinear(v_naive)

    # Case 2: use inverse model to generate voltage command
    v_cmd = predict_inverse(inverse_model, x_desired[:, None])
    x_comp = simulate_pzt_nonlinear(v_cmd)

    # Also see how well the forward model predicts x from v_cmd
    x_pred_from_forward = predict_forward(forward_model, v_cmd[:, None])

    # -----------------------------------------------------
    # Plot results
    # -----------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    ax = axes[0]
    ax.plot(t, x_desired * 1e9, label="Desired linear x(t)", lw=2)
    ax.plot(t, x_naive * 1e9, label="Naive ramp (no comp)", alpha=0.8)
    ax.plot(t, x_comp * 1e9, label="With inverse model comp", alpha=0.8)
    ax.set_ylabel("Displacement (nm)")
    ax.set_title("PZT Displacement vs Time")
    ax.grid(True)
    ax.legend()

    ax = axes[1]
    ax.plot(t, v_naive, label="Naive voltage ramp")
    ax.plot(t, v_cmd, label="Compensated voltage (inverse model)")
    ax.set_xlabel("Time (arb. units)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Voltage Profiles")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------
    # Print some error metrics
    # -----------------------------------------------------
    def rmse(a, b):
        return np.sqrt(np.mean((a - b)**2))

    err_naive = rmse(x_naive, x_desired)
    err_comp = rmse(x_comp, x_desired)
    print(f"RMSE without compensation  : {err_naive*1e9:.3f} nm")
    print(f"RMSE with compensation     : {err_comp*1e9:.3f} nm")


# =========================================================
# 5. Main: run the whole pipeline
# =========================================================

if __name__ == "__main__":
    np.random.seed(42)

    # Step 1: generate synthetic data
    print("Generating synthetic PZT + Michelson data...")
    v_all, x_all, I_all = generate_dataset(
        num_waveforms=200,  # More data for better training
        points_per_waveform=1000,
        v_min=0.0,
        v_max=10.0
    )

    # (Optional) check basic scale
    print(f"v range: {v_all.min():.2f} .. {v_all.max():.2f} V")
    print(f"x range: {x_all.min()*1e9:.2f} .. {x_all.max()*1e9:.2f} nm")

    # Step 2: train forward model v -> x
    forward_model = train_forward_model(v_all, x_all)

    # Step 3: train inverse model x -> v
    inverse_model = train_inverse_model(v_all, x_all)

    # Step 4: test linearization using inverse model
    test_linearization(forward_model, inverse_model,
                       v_min=0.0, v_max=10.0, num_points=1000)
