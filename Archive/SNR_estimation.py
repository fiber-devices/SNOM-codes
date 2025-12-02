import numpy as np
import matplotlib.pyplot as plt
import math

# =========================================================
# 1. True optical power profiles: 1-mode and 2-mode Gaussian
# =========================================================

def P_true_gaussian(x, P0, sigma_x):
    """
    Single-mode Gaussian profile vs position x.

    x       : position along scan [m], x = 0 = closest approach
    P0      : true peak power [W]
    sigma_x : spatial sigma [m]
    """
    x = np.asarray(x)
    return P0 * np.exp(-0.5 * (x / sigma_x)**2)


def P_true_two_mode_gaussian(x, P0, sigma_x, delta_x, amp_ratio=1.0):
    """
    Two polarization modes: sum of two Gaussians vs position.

    Mode 1 center at x = 0
    Mode 2 center at x = delta_x

    Before normalization:
        P_raw(x) = G1(x) + amp_ratio * G2(x)

    Then normalized so that max(P_raw) = P0, so the peak
    power is still P0.

    x         : position [m]
    P0        : target overall peak [W]
    sigma_x   : spatial sigma for both modes [m]
    delta_x   : mode separation in position [m]
    amp_ratio : relative amplitude of mode 2
    """
    x = np.asarray(x)
    G1 = np.exp(-0.5 * (x / sigma_x)**2)
    G2 = np.exp(-0.5 * ((x - delta_x) / sigma_x)**2)

    P_raw = G1 + amp_ratio * G2
    max_raw = np.max(P_raw)
    if max_raw <= 0:
        return np.zeros_like(x)
    return P0 * P_raw / max_raw


# =========================================================
# 2. Detector model: symmetric low-pass (no peak shift)
# =========================================================

def apply_detector_lowpass_symmetric(P_true, dt, bandwidth_hz, kernel_width_tau=5.0):
    """
    Zero-phase, symmetric low-pass filter to model finite detector bandwidth
    WITHOUT shifting the peak position.

    Convolves P_true with an exp(-|t|/tau) kernel, normalized.
    """
    P_true = np.asarray(P_true)
    n_sig = len(P_true)
    if n_sig < 2:
        return P_true.copy()

    tau = 1.0 / (2.0 * math.pi * bandwidth_hz)

    # Desired half-width in samples
    N_desired = int(kernel_width_tau * tau / dt)
    # Ensure kernel is at least 1 sample and not longer than signal
    N = max(1, min(N_desired, (n_sig - 1) // 2))

    t_kernel = np.arange(-N, N + 1) * dt
    h = np.exp(-np.abs(t_kernel) / tau)
    h /= h.sum()

    P_meas = np.convolve(P_true, h, mode="same")
    return P_meas


# =========================================================
# 3. Continuous scan with detector bandwidth (1 or 2 modes)
# =========================================================

def simulate_scan_with_detector(L_um, t_s, P0_W, sigma_x_um,
                                bandwidth_hz, fs=None,
                                n_modes=1,
                                delta_x_um=0.5,
                                amp_ratio=1.0):
    """
    Continuous scan from -L/2 to +L/2 in time t_s with Gaussian spatial profile,
    including detector bandwidth. Supports either 1 or 2 polarization modes.

    Parameters
    ----------
    L_um        : scan length [µm]
    t_s         : scan time [s]
    P0_W        : true overall peak power [W]
    sigma_x_um  : spatial sigma [µm]
    bandwidth_hz: detector bandwidth [Hz]
    fs          : sampling rate [Hz] (if None, choose automatically)
    n_modes     : 1 or 2
    delta_x_um  : separation between two polarization modes [µm] if n_modes=2
    amp_ratio   : relative amplitude of mode 2 (if n_modes=2)

    Returns
    -------
    t       : time array [s]
    x_um    : positions [µm]
    P_true  : true power vs time [W]
    P_det   : detector-limited power vs time [W]
    P_peak  : peak of P_det [W]
    """
    L_m = L_um * 1e-6
    sigma_x_m = sigma_x_um * 1e-6
    delta_x_m = delta_x_um * 1e-6

    if fs is None:
        fs = max(10 * bandwidth_hz, 200_000)

    n_samples = max(int(t_s * fs), 500)
    t = np.linspace(0.0, t_s, n_samples)
    dt = t[1] - t[0]

    v = L_m / t_s
    x = -L_m / 2.0 + v * t  # [m]

    # Choose envelope: 1-mode or 2-mode
    if n_modes == 1:
        P_true = P_true_gaussian(x, P0_W, sigma_x_m)
    else:
        P_true = P_true_two_mode_gaussian(x, P0_W, sigma_x_m,
                                          delta_x=delta_x_m,
                                          amp_ratio=amp_ratio)

    P_det = apply_detector_lowpass_symmetric(P_true, dt, bandwidth_hz)
    P_peak = np.max(P_det)
    x_um = x * 1e6

    return t, x_um, P_true, P_det, P_peak


# =========================================================
# 4. EOM sampling (after detector)
# =========================================================

def eom_sample_from_time_series(t, x_um, P_det, f_eom_hz, max_points=2000):
    """
    Sample the detector output at times corresponding to an EOM clock
    at f_eom_hz.

    t        : time array [s] (monotonic)
    x_um     : position array [µm]
    P_det    : detector output vs time [W]
    f_eom_hz : EOM frequency [Hz]
    max_points : maximum number of EOM samples to keep (for plotting)

    Returns
    -------
    t_eom     : EOM sample times [s]
    x_eom_um  : sampled positions [µm]
    P_eom     : sampled detector power [W]
    """
    t = np.asarray(t)
    x_um = np.asarray(x_um)
    P_det = np.asarray(P_det)

    t_end = t[-1]
    N_cycles = int(np.floor(f_eom_hz * t_end))

    if N_cycles < 1:
        return np.array([]), np.array([]), np.array([])

    t_eom = np.arange(0, N_cycles + 1) / f_eom_hz

    if len(t_eom) > max_points:
        stride = int(np.ceil(len(t_eom) / max_points))
        t_eom = t_eom[::stride]

    t_eom = np.clip(t_eom, t[0], t_end)

    x_eom_um = np.interp(t_eom, t, x_um)
    P_eom = np.interp(t_eom, t, P_det)

    return t_eom, x_eom_um, P_eom


# =========================================================
# 5. Wrapper: reference peak + arbitrary scan + EOM + birefringence
# =========================================================

def estimate_scan_with_eom(L_um, t_s,
                           P_ref_W=75e-9,
                           L_ref_um=20.0,
                           t_ref_s=0.25,
                           sigma_x_um=0.3,
                           bandwidth_hz=1e5,
                           f_eom_hz=100e6,
                           fs=None,
                           max_points=2000,
                           # birefringence / polarization split:
                           f_pol_split_hz=None,
                           pol_match_tol=0.01,
                           delta_x_pol_um=0.5,
                           pol_amp_ratio=1.0):
    """
    Use reference slow scan (20 µm, 0.25 s, 75 nW peak) as true peak P0.

    Then simulate scan (L_um, t_s) with:
      - detector bandwidth
      - EOM sampling
      - optional second polarization mode when f_eom ~ f_pol_split

    Parameters
    ----------
    L_um, t_s         : scan length/time
    P_ref_W           : reference peak [W] (e.g., 75 nW)
    sigma_x_um        : spatial sigma [µm]
    bandwidth_hz      : detector bandwidth [Hz]
    f_eom_hz          : EOM frequency [Hz]
    f_pol_split_hz    : polarization mode splitting [Hz]
    pol_match_tol     : fractional tolerance; if
                        |f_eom - f_pol_split| <= pol_match_tol * f_pol_split
                        → use two-mode Gaussian
    delta_x_pol_um    : separation between the two polarization peaks [µm]
    pol_amp_ratio     : relative amplitude of 2nd mode

    Returns
    -------
    x_um      : continuous positions [µm]
    P_true    : true power vs x [W]
    P_det     : detector-limited power vs x [W]
    P_peak    : peak of P_det [W]
    x_eom_um  : EOM-sampled positions [µm]
    P_eom     : EOM-sampled power [W]
    n_modes   : 1 or 2 (how many polarization modes used)
    """
    P0_W = P_ref_W

    # Decide whether to use 1 or 2 polarization modes
    n_modes = 1
    if (f_pol_split_hz is not None) and (f_pol_split_hz > 0):
        if abs(f_eom_hz - f_pol_split_hz) <= pol_match_tol * f_pol_split_hz:
            n_modes = 2

    # Continuous scan with detector bandwidth
    t, x_um, P_true, P_det, P_peak = simulate_scan_with_detector(
        L_um=L_um,
        t_s=t_s,
        P0_W=P0_W,
        sigma_x_um=sigma_x_um,
        bandwidth_hz=bandwidth_hz,
        fs=fs,
        n_modes=n_modes,
        delta_x_um=delta_x_pol_um,
        amp_ratio=pol_amp_ratio,
    )

    # EOM sampling (after detector)
    t_eom, x_eom_um, P_eom = eom_sample_from_time_series(
        t, x_um, P_det,
        f_eom_hz=f_eom_hz,
        max_points=max_points
    )

    return x_um, P_true, P_det, P_peak, x_eom_um, P_eom, n_modes


# =========================================================
# 6. Example usage
# =========================================================

if __name__ == "__main__":
    # Base parameters
    P_ref_W = 75e-9      # 75 nW
    L_um = 20.0          # scan length
    sigma_x_um = 0.3     # mode width ~ fiber radius
    bandwidth_hz = 1e5   # detector bandwidth
    f_eom_hz = 100e6     # EOM frequency

    # Polarization splitting (example): 100 MHz
    f_pol_split_hz = 100e6
    delta_x_pol_um = 0.5  # separation between pol peaks in position
    pol_amp_ratio = 1.0   # equal heights

    # 1) EOM NOT matched to splitting (single peak)
    t_s = 0.25  # slow scan
    f_eom_off = 80e6

    x_off_um, P_true_off, P_det_off, P_peak_off, \
        x_eom_off_um, P_eom_off, n_modes_off = estimate_scan_with_eom(
            L_um=L_um,
            t_s=t_s,
            P_ref_W=P_ref_W,
            L_ref_um=20.0,
            t_ref_s=0.25,
            sigma_x_um=sigma_x_um,
            bandwidth_hz=bandwidth_hz,
            f_eom_hz=f_eom_off,
            fs=None,
            max_points=2000,
            f_pol_split_hz=f_pol_split_hz,
            pol_match_tol=0.01,
            delta_x_pol_um=delta_x_pol_um,
            pol_amp_ratio=pol_amp_ratio
        )

    print("=== Case 1: EOM off resonance with pol splitting ===")
    print(f"f_eom = {f_eom_off/1e6:.1f} MHz, n_modes = {n_modes_off}")
    print(f"P_peak ≈ {P_peak_off*1e9:.2f} nW")
    print()

    # 2) EOM matched to splitting (two peaks)
    f_eom_on = f_pol_split_hz

    x_on_um, P_true_on, P_det_on, P_peak_on, \
        x_eom_on_um, P_eom_on, n_modes_on = estimate_scan_with_eom(
            L_um=L_um,
            t_s=t_s,
            P_ref_W=P_ref_W,
            L_ref_um=20.0,
            t_ref_s=0.25,
            sigma_x_um=sigma_x_um,
            bandwidth_hz=bandwidth_hz,
            f_eom_hz=f_eom_on,
            fs=None,
            max_points=2000,
            f_pol_split_hz=f_pol_split_hz,
            pol_match_tol=0.01,
            delta_x_pol_um=delta_x_pol_um,
            pol_amp_ratio=pol_amp_ratio
        )

    print("=== Case 2: EOM on resonance with pol splitting ===")
    print(f"f_eom = {f_eom_on/1e6:.1f} MHz, n_modes = {n_modes_on}")
    print(f"P_peak ≈ {P_peak_on*1e9:.2f} nW")
    print()

    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    # Case 1: off resonance → single peak
    ax = axes[0]
    ax.plot(x_off_um, P_det_off*1e9, label="Detector (continuous)")
    ax.scatter(x_eom_off_um, P_eom_off*1e9,
               s=8, alpha=0.6, label="EOM samples")
    ax.set_title(f"Off-resonance EOM (single pol mode): f_eom = {f_eom_off/1e6:.1f} MHz")
    ax.set_ylabel("Power (nW)")
    ax.grid(True)
    ax.legend()

    # Case 2: on resonance → two peaks
    ax = axes[1]
    ax.plot(x_on_um, P_det_on*1e9, label="Detector (continuous)")
    ax.scatter(x_eom_on_um, P_eom_on*1e9,
               s=8, alpha=0.6, label="EOM samples")
    ax.set_title(f"On-resonance EOM (two pol modes): f_eom = {f_eom_on/1e6:.1f} MHz")
    ax.set_xlabel("Scan position (µm)")
    ax.set_ylabel("Power (nW)")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()
