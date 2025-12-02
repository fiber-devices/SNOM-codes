#!/usr/bin/env python3
"""
PZT Motion Linearization using Machine Learning and Michelson Interferometer Feedback

This simulation:
1. Models nonlinear PZT motion with hysteresis and creep
2. Simulates Michelson interferometer signals
3. Uses ML to learn the PZT nonlinear behavior
4. Derives compensated voltage profiles for linear motion
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# 1. PZT NONLINEAR MOTION MODEL
# =============================================================================

class PZTModel:
    """
    Models PZT actuator with realistic nonlinearities:
    - Hysteresis (using Preisach-like model)
    - Creep
    - Nonlinear gain
    - Phase noise
    """
    
    def __init__(self, 
                 max_displacement=10e-6,  # 10 microns max displacement
                 hysteresis_factor=0.15,   # 15% hysteresis
                 creep_factor=0.05,        # 5% creep
                 nonlinearity_order=3,     # Polynomial nonlinearity
                 phase_noise_std=0.02):    # Phase noise standard deviation
        
        self.max_displacement = max_displacement
        self.hysteresis_factor = hysteresis_factor
        self.creep_factor = creep_factor
        self.nonlinearity_order = nonlinearity_order
        self.phase_noise_std = phase_noise_std
        
        # Internal state for hysteresis
        self.prev_voltage = 0
        self.prev_displacement = 0
        self.voltage_direction = 1
        
    def reset(self):
        """Reset internal state"""
        self.prev_voltage = 0
        self.prev_displacement = 0
        self.voltage_direction = 1
        
    def apply_hysteresis(self, voltage, displacement_linear):
        """Apply hysteresis effect using simplified Preisach model"""
        # Determine voltage direction
        if voltage > self.prev_voltage:
            new_direction = 1
        elif voltage < self.prev_voltage:
            new_direction = -1
        else:
            new_direction = self.voltage_direction
            
        # Hysteresis offset depends on direction
        hysteresis_offset = self.hysteresis_factor * self.max_displacement * new_direction * 0.5
        
        # Smooth transition
        alpha = 0.3  # Smoothing factor
        displacement_hyst = displacement_linear + hysteresis_offset * np.tanh(3 * voltage)
        
        # Update state
        self.prev_voltage = voltage
        self.voltage_direction = new_direction
        
        return displacement_hyst
    
    def apply_nonlinearity(self, voltage):
        """Apply polynomial nonlinearity to voltage-displacement relationship"""
        # Normalized voltage (0 to 1)
        v_norm = voltage / 150.0  # Assuming 150V max
        
        # Polynomial nonlinearity (dominant linear + higher order terms)
        displacement = (0.85 * v_norm + 
                       0.10 * v_norm**2 + 
                       0.05 * v_norm**3 - 
                       0.02 * v_norm**4)
        
        return displacement * self.max_displacement
    
    def apply_creep(self, displacement, time_step, total_time):
        """Apply logarithmic creep effect"""
        if total_time > 0:
            creep = self.creep_factor * displacement * np.log(1 + total_time / 0.1)
            return displacement + creep * (1 - np.exp(-time_step / 0.5))
        return displacement
    
    def get_displacement(self, voltage, time_step=0.001, total_time=0):
        """
        Get actual displacement for given voltage including all nonlinearities
        """
        # Apply nonlinear voltage-displacement relationship
        displacement_nonlin = self.apply_nonlinearity(voltage)
        
        # Apply hysteresis
        displacement_hyst = self.apply_hysteresis(voltage, displacement_nonlin)
        
        # Apply creep
        displacement_creep = self.apply_creep(displacement_hyst, time_step, total_time)
        
        # Add phase noise
        phase_noise = np.random.normal(0, self.phase_noise_std * self.max_displacement)
        
        final_displacement = displacement_creep + phase_noise
        
        self.prev_displacement = final_displacement
        
        return final_displacement
    
    def get_displacement_array(self, voltages, dt=0.001):
        """Get displacement array for voltage array"""
        self.reset()
        displacements = []
        
        for i, v in enumerate(voltages):
            d = self.get_displacement(v, dt, i * dt)
            displacements.append(d)
            
        return np.array(displacements)


# =============================================================================
# 2. MICHELSON INTERFEROMETER MODEL
# =============================================================================

class MichelsonInterferometer:
    """
    Simulates Michelson interferometer signal from PZT displacement
    """
    
    def __init__(self, 
                 wavelength=632.8e-9,  # HeNe laser wavelength
                 visibility=0.95,       # Fringe visibility
                 intensity_noise=0.02): # Intensity noise level
        
        self.wavelength = wavelength
        self.visibility = visibility
        self.intensity_noise = intensity_noise
        self.k = 4 * np.pi / wavelength  # Wave number (factor of 4 for double pass)
        
    def get_signal(self, displacement):
        """
        Get interferometer signal for given displacement
        I = I0 * (1 + V * cos(k * d + phi))
        """
        # Base intensity
        I0 = 1.0
        
        # Phase from displacement
        phase = self.k * displacement
        
        # Interferometer signal
        signal = I0 * (1 + self.visibility * np.cos(phase))
        
        # Add intensity noise
        noise = np.random.normal(0, self.intensity_noise)
        
        return signal + noise
    
    def get_signal_array(self, displacements):
        """Get signal array for displacement array"""
        return np.array([self.get_signal(d) for d in displacements])
    
    def unwrap_phase(self, signals):
        """
        Extract displacement from interferometer signals using phase unwrapping
        """
        # Normalize signals
        signals_norm = (signals - np.mean(signals)) / (np.max(signals) - np.min(signals)) * 2
        signals_norm = np.clip(signals_norm, -0.999, 0.999)
        
        # Get wrapped phase
        phase_wrapped = np.arccos(signals_norm / self.visibility)
        
        # Simple phase unwrapping
        phase_unwrapped = np.unwrap(phase_wrapped)
        
        # Convert to displacement
        displacement = phase_unwrapped / self.k
        
        return displacement


# =============================================================================
# 3. MACHINE LEARNING MODEL FOR PZT CHARACTERIZATION
# =============================================================================

class PZTLearner:
    """
    ML model to learn PZT nonlinear behavior and derive compensation
    """
    
    def __init__(self):
        self.forward_model = None  # Voltage -> Displacement
        self.inverse_model = None  # Desired Displacement -> Required Voltage
        self.scaler_v = StandardScaler()
        self.scaler_d = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, voltages):
        """
        Create feature matrix including voltage history and derivatives
        """
        n = len(voltages)
        features = np.zeros((n, 7))
        
        # Current voltage
        features[:, 0] = voltages
        
        # Voltage squared and cubed (for nonlinearity)
        features[:, 1] = voltages**2
        features[:, 2] = voltages**3
        
        # Voltage derivative (for hysteresis direction)
        features[1:, 3] = np.diff(voltages)
        features[0, 3] = features[1, 3]
        
        # Second derivative
        features[2:, 4] = np.diff(voltages, 2)
        features[:2, 4] = features[2, 4]
        
        # Previous voltage (for hysteresis)
        features[1:, 5] = voltages[:-1]
        features[0, 5] = voltages[0]
        
        # Cumulative voltage (for creep)
        features[:, 6] = np.cumsum(np.abs(np.diff(np.concatenate([[0], voltages]))))
        
        return features
    
    def train(self, voltages, displacements, test_size=0.2):
        """
        Train ML models on PZT data
        """
        print("Training ML models...")
        
        # Prepare features
        X = self.prepare_features(voltages)
        y = displacements.reshape(-1, 1)
        
        # Scale features
        X_scaled = self.scaler_v.fit_transform(X)
        y_scaled = self.scaler_d.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, random_state=42
        )
        
        # Train forward model (Voltage -> Displacement)
        self.forward_model = MLPRegressor(
            hidden_layer_sizes=(128, 256, 128, 64),
            activation='tanh',
            solver='adam',
            max_iter=3000,
            early_stopping=True,
            validation_fraction=0.1,
            learning_rate='adaptive',
            random_state=42
        )
        self.forward_model.fit(X_train, y_train.ravel())
        
        # Evaluate forward model
        train_score = self.forward_model.score(X_train, y_train.ravel())
        test_score = self.forward_model.score(X_test, y_test.ravel())
        print(f"Forward model - Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
        
        # Train inverse model (Displacement -> Voltage)
        # Use displacement with context as input
        # Create inverse training data with displacement-based features
        y_inv = X_train[:, 0]  # Voltage is target
        
        self.inverse_model = MLPRegressor(
            hidden_layer_sizes=(128, 256, 128, 64),
            activation='tanh',
            solver='adam',
            max_iter=3000,
            early_stopping=True,
            validation_fraction=0.1,
            learning_rate='adaptive',
            random_state=42
        )
        
        # For inverse model, use displacement as input and voltage as output
        self.inverse_model.fit(y_train, y_inv)
        
        inv_train_score = self.inverse_model.score(y_train, X_train[:, 0])
        inv_test_score = self.inverse_model.score(y_test, X_test[:, 0])
        print(f"Inverse model - Train R²: {inv_train_score:.4f}, Test R²: {inv_test_score:.4f}")
        
        self.is_trained = True
        
        return test_score, inv_test_score
    
    def predict_displacement(self, voltages):
        """Predict displacement from voltage using forward model"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
            
        X = self.prepare_features(voltages)
        X_scaled = self.scaler_v.transform(X)
        y_scaled = self.forward_model.predict(X_scaled)
        y = self.scaler_d.inverse_transform(y_scaled.reshape(-1, 1))
        
        return y.ravel()
    
    def compute_compensation_voltage(self, desired_displacements, initial_voltage_profile):
        """
        Compute compensated voltage profile for desired linear displacement
        Uses direct inverse model prediction followed by iterative refinement
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        print("Computing compensation voltage profile...")
        
        n = len(desired_displacements)
        
        # Step 1: Use inverse model for initial guess
        d_scaled = self.scaler_d.transform(desired_displacements.reshape(-1, 1))
        v_initial_scaled = self.inverse_model.predict(d_scaled)
        
        # Convert back to original scale
        # The inverse model predicts scaled voltage values
        v_range = self.scaler_v.scale_[0]
        v_mean = self.scaler_v.mean_[0]
        compensated_voltage = v_initial_scaled * v_range + v_mean
        compensated_voltage = np.clip(compensated_voltage, 0, 150)
        
        # Ensure smooth start from zero - add small ramp at beginning if needed
        # This prevents flat region at start
        if compensated_voltage[0] < 1.0 and len(compensated_voltage) > 10:
            # If starting voltage is very low, ensure smooth ramp up
            ramp_length = min(100, int(0.05 * n))  # First 50ms or 100 points
            if ramp_length > 1:
                initial_val = max(0, compensated_voltage[0])
                target_val = compensated_voltage[ramp_length]
                compensated_voltage[:ramp_length] = np.linspace(initial_val, target_val, ramp_length)
        
        # Step 2: Iterative refinement
        best_voltage = compensated_voltage.copy()
        best_error = float('inf')
        
        learning_rate = 0.5
        momentum = np.zeros_like(compensated_voltage)
        beta = 0.7
        
        for iteration in range(100):
            # Predict displacement with current voltage
            predicted_displacement = self.predict_displacement(compensated_voltage)
            
            # Compute error
            error = desired_displacements - predicted_displacement
            rms_error = np.sqrt(np.mean(error**2))
            
            # Track best solution
            if rms_error < best_error:
                best_error = rms_error
                best_voltage = compensated_voltage.copy()
            
            # Estimate local sensitivity
            dv = 0.5
            voltages_plus = np.clip(compensated_voltage + dv, 0, 150)
            disp_plus = self.predict_displacement(voltages_plus)
            sensitivity = (disp_plus - predicted_displacement) / dv
            
            # Robust sensitivity clipping
            sensitivity = np.clip(sensitivity, 1e-9, None)
            
            # Compute voltage correction with momentum
            voltage_correction = error / sensitivity
            momentum = beta * momentum + (1 - beta) * voltage_correction
            
            # Apply correction
            compensated_voltage = compensated_voltage + learning_rate * momentum
            compensated_voltage = np.clip(compensated_voltage, 0, 150)
            
            # Smooth after initial iterations, but preserve the initial region
            if iteration > 10:
                # Don't smooth the very beginning to avoid flat region
                smooth_start_idx = max(50, int(0.1 * n))  # Start smoothing after ~0.1s
                if smooth_start_idx < n:
                    compensated_voltage[smooth_start_idx:] = savgol_filter(
                        compensated_voltage[smooth_start_idx:], 15, 3
                    )
            
            # Adaptive learning rate
            if iteration > 30:
                learning_rate = 0.3
            if iteration > 60:
                learning_rate = 0.1
            
            # Progress report
            if (iteration + 1) % 20 == 0:
                print(f"  Iteration {iteration+1}: RMS error = {rms_error*1e6:.3f} µm")
            
            # Early stopping
            if rms_error < 5e-9:
                print(f"  Converged at iteration {iteration+1}")
                break
        
        # Return best solution found
        final_pred = self.predict_displacement(best_voltage)
        final_error = np.sqrt(np.mean((desired_displacements - final_pred)**2))
        print(f"  Final best RMS error: {final_error*1e6:.3f} µm")
        
        return best_voltage


# =============================================================================
# 4. MAIN SIMULATION AND VISUALIZATION
# =============================================================================

def run_simulation():
    """
    Main simulation routine
    """
    print("=" * 60)
    print("PZT LINEARIZATION SIMULATION")
    print("=" * 60)
    
    # Create PZT and interferometer models
    pzt = PZTModel(
        max_displacement=10e-6,
        hysteresis_factor=0.15,
        creep_factor=0.05,
        phase_noise_std=0.002  # Reduced noise for clearer demonstration
    )
    
    interferometer = MichelsonInterferometer(
        wavelength=632.8e-9,
        visibility=0.95,
        intensity_noise=0.02
    )
    
    # Time parameters
    dt = 0.001  # 1 ms time step
    t_total = 2.0  # 2 seconds
    t = np.arange(0, t_total, dt)
    n_points = len(t)
    
    print(f"\nSimulation parameters:")
    print(f"  Time step: {dt*1000:.1f} ms")
    print(f"  Total time: {t_total:.1f} s")
    print(f"  Number of points: {n_points}")
    
    # ==========================================================================
    # Step 1: Generate training data with various voltage profiles
    # ==========================================================================
    print("\n" + "-" * 60)
    print("Step 1: Generating training data...")
    
    # Multiple training profiles
    training_voltages = []
    training_displacements = []
    
    # Profile 1: Triangular wave (tests hysteresis)
    v_triangle = 75 + 75 * np.abs(2 * (t / 0.5 - np.floor(t / 0.5 + 0.5)))
    training_voltages.append(v_triangle)
    
    # Profile 2: Sinusoidal
    v_sine = 75 + 75 * np.sin(2 * np.pi * t / 1.0)
    training_voltages.append(v_sine)
    
    # Profile 3: Step function (tests creep)
    v_step = np.where(t < 1.0, 50, 100)
    training_voltages.append(v_step)
    
    # Profile 4: Ramp up
    v_ramp = 150 * t / t_total
    training_voltages.append(v_ramp)
    
    # Profile 5: Ramp down
    v_ramp_down = 150 * (1 - t / t_total)
    training_voltages.append(v_ramp_down)
    
    # Profile 6: Slow ramp (to capture true linear behavior)
    v_slow_ramp = 120 * t / t_total
    training_voltages.append(v_slow_ramp)
    
    # Profile 7: Multiple frequency sine
    v_multi = 75 + 40 * np.sin(2 * np.pi * t * 0.5) + 35 * np.sin(2 * np.pi * t * 2)
    training_voltages.append(v_multi)
    
    # Profile 8: Staircase
    n_steps = 10
    v_stair = np.zeros(n_points)
    for i in range(n_steps):
        start = int(i * n_points / n_steps)
        end = int((i + 1) * n_points / n_steps)
        v_stair[start:end] = i * 150 / (n_steps - 1)
    training_voltages.append(v_stair)
    
    # Profile 9: Random walk
    v_random = 75 + np.cumsum(np.random.randn(n_points)) * 0.5
    v_random = np.clip(v_random, 0, 150)
    training_voltages.append(v_random)
    
    # Generate displacements for all profiles
    for v_profile in training_voltages:
        pzt.reset()
        d = pzt.get_displacement_array(v_profile, dt)
        training_displacements.append(d)
    
    # Combine all training data
    all_voltages = np.concatenate(training_voltages)
    all_displacements = np.concatenate(training_displacements)
    
    print(f"  Generated {len(training_voltages)} training profiles")
    print(f"  Total training points: {len(all_voltages)}")
    
    # ==========================================================================
    # Step 2: Train ML model
    # ==========================================================================
    print("\n" + "-" * 60)
    print("Step 2: Training ML model...")
    
    learner = PZTLearner()
    forward_score, inverse_score = learner.train(all_voltages, all_displacements)
    
    # ==========================================================================
    # Step 3: Test with linear ramp
    # ==========================================================================
    print("\n" + "-" * 60)
    print("Step 3: Testing linearization...")
    
    # Desired linear displacement (start from small non-zero to avoid flat region)
    d_min = 0.1e-6  # Start from 0.1 microns instead of 0
    d_max = 8e-6  # 8 microns
    desired_displacement = np.linspace(d_min, d_max, n_points)
    
    # Initial voltage guess (linear mapping)
    initial_voltage = 150 * desired_displacement / pzt.max_displacement
    
    # Get actual displacement with uncompensated voltage
    pzt.reset()
    uncompensated_displacement = pzt.get_displacement_array(initial_voltage, dt)
    
    # Compute compensated voltage
    compensated_voltage = learner.compute_compensation_voltage(
        desired_displacement, initial_voltage
    )
    
    # Get actual displacement with compensated voltage
    pzt.reset()
    compensated_displacement = pzt.get_displacement_array(compensated_voltage, dt)
    
    # ==========================================================================
    # Step 4: Analyze results
    # ==========================================================================
    print("\n" + "-" * 60)
    print("Step 4: Analyzing results...")
    
    # Calculate linearity errors
    error_uncompensated = uncompensated_displacement - desired_displacement
    error_compensated = compensated_displacement - desired_displacement
    
    rms_uncompensated = np.sqrt(np.mean(error_uncompensated**2))
    rms_compensated = np.sqrt(np.mean(error_compensated**2))
    
    max_uncompensated = np.max(np.abs(error_uncompensated))
    max_compensated = np.max(np.abs(error_compensated))
    
    improvement = (1 - rms_compensated / rms_uncompensated) * 100
    
    print(f"\nLinearity Analysis:")
    print(f"  Uncompensated RMS error: {rms_uncompensated*1e6:.3f} µm")
    print(f"  Compensated RMS error:   {rms_compensated*1e6:.3f} µm")
    print(f"  Improvement: {improvement:.1f}%")
    print(f"\n  Uncompensated max error: {max_uncompensated*1e6:.3f} µm")
    print(f"  Compensated max error:   {max_compensated*1e6:.3f} µm")
    
    # ==========================================================================
    # Step 5: Simulate interferometer measurement
    # ==========================================================================
    print("\n" + "-" * 60)
    print("Step 5: Simulating interferometer measurements...")
    
    # Get interferometer signals
    mi_signal_uncompensated = interferometer.get_signal_array(uncompensated_displacement)
    mi_signal_compensated = interferometer.get_signal_array(compensated_displacement)
    
    # ==========================================================================
    # Step 6: Create visualization
    # ==========================================================================
    print("\n" + "-" * 60)
    print("Step 6: Creating visualizations...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Voltage profiles
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(t, initial_voltage, 'b-', label='Uncompensated', alpha=0.7)
    ax1.plot(t, compensated_voltage, 'r-', label='Compensated', alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('Voltage Profiles')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Displacement comparison
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(t, desired_displacement * 1e6, 'k--', label='Desired (Linear)', linewidth=2)
    ax2.plot(t, uncompensated_displacement * 1e6, 'b-', label='Uncompensated', alpha=0.7)
    ax2.plot(t, compensated_displacement * 1e6, 'r-', label='Compensated', alpha=0.7)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Displacement (µm)')
    ax2.set_title('Displacement Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Linearity error
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(t, error_uncompensated * 1e6, 'b-', label='Uncompensated', alpha=0.7)
    ax3.plot(t, error_compensated * 1e6, 'r-', label='Compensated', alpha=0.7)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Error (µm)')
    ax3.set_title('Linearity Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Voltage-Displacement relationship (Hysteresis)
    ax4 = fig.add_subplot(3, 2, 4)
    # Use triangular profile to show hysteresis
    pzt.reset()
    v_test = training_voltages[0][:1000]  # First half of triangle
    d_test = pzt.get_displacement_array(v_test, dt)
    ax4.plot(v_test, d_test * 1e6, 'b-', alpha=0.7)
    ax4.set_xlabel('Voltage (V)')
    ax4.set_ylabel('Displacement (µm)')
    ax4.set_title('Voltage-Displacement Hysteresis Loop')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Interferometer signals
    ax5 = fig.add_subplot(3, 2, 5)
    # Show full 2 seconds
    ax5.plot(t, mi_signal_compensated, 'r-', alpha=0.7)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Intensity (a.u.)')
    ax5.set_title('Michelson Interferometer Signal (Compensated)')
    ax5.set_xlim(0, 2.0)  # Set x-limit to 2 seconds
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Error histogram
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.hist(error_uncompensated * 1e9, bins=50, alpha=0.5, label='Uncompensated', color='blue')
    ax6.hist(error_compensated * 1e9, bins=50, alpha=0.5, label='Compensated', color='red')
    ax6.set_xlabel('Error (nm)')
    ax6.set_ylabel('Count')
    ax6.set_title('Error Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = '/mnt/user-data/outputs/pzt_linearization_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nResults saved to: {output_path}")
    
    # ==========================================================================
    # Step 7: Save compensation data
    # ==========================================================================
    print("\n" + "-" * 60)
    print("Step 7: Saving compensation data...")
    
    # Save voltage compensation profile
    compensation_data = np.column_stack([
        t, 
        desired_displacement * 1e6, 
        initial_voltage, 
        compensated_voltage,
        compensated_voltage - initial_voltage
    ])
    
    header = "Time(s), Desired_Displacement(um), Initial_Voltage(V), Compensated_Voltage(V), Voltage_Correction(V)"
    data_path = '/mnt/user-data/outputs/pzt_compensation_data.csv'
    np.savetxt(data_path, compensation_data, delimiter=',', header=header, comments='')
    print(f"Compensation data saved to: {data_path}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"\nKey Results:")
    print(f"  • ML Forward Model R²: {forward_score:.4f}")
    print(f"  • ML Inverse Model R²: {inverse_score:.4f}")
    print(f"  • Linearity improvement: {improvement:.1f}%")
    print(f"  • Final RMS error: {rms_compensated*1e9:.1f} nm")
    
    return {
        'forward_score': forward_score,
        'inverse_score': inverse_score,
        'rms_uncompensated': rms_uncompensated,
        'rms_compensated': rms_compensated,
        'improvement': improvement,
        'compensated_voltage': compensated_voltage,
        'learner': learner
    }


if __name__ == "__main__":
    results = run_simulation()
