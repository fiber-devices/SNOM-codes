import os
import json
import time
import datetime
from typing import Sequence, Dict, Any, Optional
from collections import OrderedDict

import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, WAIT_INFINITELY
from SNOM_stage import FiberHeadController

# Global controller instance (will be initialized when needed)
_fiber_controller = None

def get_controller():
    """Get or create the global FiberHeadController instance."""
    global _fiber_controller
    if _fiber_controller is None:
        _fiber_controller = FiberHeadController()
    return _fiber_controller

def piezo_getpos(pi_serialnum: str = "") -> Dict[str, float]:
    """
    Get current position from SNOM stage controller.
    Axis mapping: x -> x_fine, y -> y_fine, z -> z_coarse

    Parameters
    ----------
    pi_serialnum : str
        (Unused - kept for compatibility)

    Returns
    -------
    dict: {"x": float, "y": float, "z": float}
        Current position [µm]
    """
    ctrl = get_controller()
    
    # Get positions with direct mapping
    x_pos = ctrl.get_position('x_fine')  # µm
    y_pos = ctrl.get_position('y_fine')  # µm
    z_pos = ctrl.get_position('z_coarse')  # µm
    
    return {"x": float(x_pos), "y": float(y_pos), "z": float(z_pos)}

def piezo_move(
    *,
    target_x: Optional[float] = None,
    target_y: Optional[float] = None,
    target_z: Optional[float] = None,
    pi_serialnum: str = "",) -> Dict[str, float]:
    """
    Move SNOM stage to specified position (µm). Unmoved axes maintain current position.
    Axis mapping: x -> x_fine, y -> y_fine, z -> z_coarse

    Parameters
    ----------
    target_x, target_y, target_z : float or None
        Target position [µm]. None = don't move that axis.
    pi_serialnum : str
        (Unused - kept for compatibility)

    Returns
    -------
    dict: {"x": float, "y": float, "z": float}
        Actual position after movement [µm].
    """
    ctrl = get_controller()
    
    # Move axes with direct mapping (only if target is specified)
    if target_x is not None:
        ctrl.x(target_x, wait=True)  # x -> x_fine
    
    if target_y is not None:
        ctrl.y(target_y, wait=True)  # y -> y_fine
    
    if target_z is not None:
        ctrl.z(target_z, wait=True)  # z -> z_coarse
    
    # Return actual positions
    return piezo_getpos()


def counter(
    ai_channel: str = "cDAQ1Mod1/ai0",
    sample_rate: float = 10_000,
    num_samples: int = 2_000,
    window_size: int = 100,
    update_interval: float = 0.1,
    title: str = "Real-time Averaged Voltage") -> None:
    """
    Acquire analog voltage via continuous sampling and average every num_samples.
    Live plotting removed. Only latest window_size average values are kept,
    and statistics are output to stdout every update_interval. Exit with Ctrl+C.

    Parameters
    ----------
    ai_channel : str        Example: "cDAQ1Mod1/ai0"
    sample_rate : float     Sampling rate [Hz]
    num_samples : int       Number of samples for averaging
    window_size : int       Number of average values to keep in internal buffer
    update_interval : float Log output interval [s]
    title : str             (Unused - for compatibility)
    """
    task = nidaqmx.Task()
    task.ai_channels.add_ai_voltage_chan(ai_channel)
    task.timing.cfg_samp_clk_timing(
        rate=sample_rate,
        sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=num_samples
    )
    task.start()

    data = []
    last_log = 0.0
    t0 = time.perf_counter()

    try:
        while True:
            block = task.read(
                number_of_samples_per_channel=num_samples,
                timeout=max(2.0, 2.5 * num_samples / sample_rate)
            )
            avg_v = float(np.mean(np.asarray(block, dtype=float)))
            data.append(avg_v)
            if len(data) > window_size:
                data = data[-window_size:]

            now = time.perf_counter()
            if now - last_log >= update_interval:
                arr = np.asarray(data, dtype=float)
                print(
                    f"[{now - t0:8.2f}s] avg={arr[-1]: .6f} V  "
                    f"(win n={len(arr)}, mean={arr.mean(): .6f}, std={arr.std(ddof=1) if len(arr)>1 else 0.0: .6f})"
                )
                last_log = now

    except KeyboardInterrupt:
        print("Manually stopped.")

    finally:
        try:
            task.stop()
            task.close()
        except Exception:
            pass


def measure_piezo_sweep(
    *,
    ID: str = "CRYb195",
    scan_axis: str = "x",                   # "x" | "y" | "z"
    scan_values: Sequence[float] = tuple(np.linspace(5, 10, 501)),
    const_x: float = 0.0,
    const_y: float = 0.0,
    const_z: float = 0.0,
    sample_rate: float = 100_000.0,
    num_samples: int = 1000,
    ai_physical: str = "cDAQ1Mod1/ai0:3",
    channel_names: Sequence[str] = ("Probe", "Transmission", "Interferometer", "Feedback"),
    settle_sec: float = 0.0,
    progress_every: int = 50,
    save_root: str = "data",
    pi_serialnum: str = "",
    return_data: bool = False,
    plot_every_points: int = 50,            # Unused (kept for compatibility)
    n_rep: int = 1,) -> Dict[str, Any]:
    """
    Sweep a single axis (x, y, z) and acquire/save raw waveforms (npz).
    Axis mapping: x -> x_fine, y -> y_fine, z -> z_coarse
    - Can repeat measurement n_rep times
    - Live plotting removed
    """
    if scan_axis not in ["x", "y", "z"]:
        raise ValueError("scan_axis must be 'x','y','z'")
    scan_values = np.asarray(scan_values, dtype=float)
    if scan_values.ndim != 1 or len(scan_values) == 0:
        raise ValueError("scan_values must be a 1D sequence with length > 0")

    n_steps = len(scan_values)
    n_rep = int(max(1, n_rep))

    # Save destination
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(save_root, f"{ts}_{ID}")
    os.makedirs(save_dir, exist_ok=True)
    base_filename = f"{ts}_{ID}_{scan_axis}"
    npz_path = os.path.join(save_dir, f"{base_filename}.npz")
    meta_path = os.path.join(save_dir, "meta.json")

    # DAQ
    task = nidaqmx.Task()
    task.ai_channels.add_ai_voltage_chan(ai_physical)
    task.timing.cfg_samp_clk_timing(
        rate=sample_rate,
        sample_mode=AcquisitionType.FINITE,
        samps_per_chan=num_samples
    )

    # Get SNOM stage controller
    ctrl = get_controller()

    # Buffers
    positions = np.empty((n_rep, n_steps, 3), dtype=np.float32)
    waveforms = np.empty((n_rep, n_steps, 4, num_samples), dtype=np.float32)

    def _acquire_block(task_, N, fs, timeout=None):
        if timeout is None:
            timeout = max(2.0, 2.5 * (N / float(fs)))
        task_.start()
        try:
            data = task_.read(number_of_samples_per_channel=N, timeout=timeout)
        finally:
            try:
                task_.stop()
            except Exception:
                pass
        return np.asarray(data, dtype=float)

    # ---- Repetition loop ----
    t0 = time.perf_counter()
    try:
        for rep in range(n_rep):
            # Set initial position (constant values + first scan value)
            pos_dict = {"x": float(const_x), "y": float(const_y), "z": float(const_z)}
            pos_dict[scan_axis] = float(scan_values[0])
            
            # Move to initial position using new hardware
            piezo_move(target_x=pos_dict["x"], target_y=pos_dict["y"], target_z=pos_dict["z"])

            for idx, val in enumerate(scan_values):
                pos_dict[scan_axis] = float(val)
                
                # Move scan axis only (others stay constant)
                if scan_axis == "x":
                    piezo_move(target_x=pos_dict["x"])
                elif scan_axis == "y":
                    piezo_move(target_y=pos_dict["y"])
                elif scan_axis == "z":
                    piezo_move(target_z=pos_dict["z"])
                
                if settle_sec > 0:
                    time.sleep(settle_sec)

                # Get actual positions
                pos = piezo_getpos()
                positions[rep, idx] = (pos["x"], pos["y"], pos["z"])

                samples = _acquire_block(task, num_samples, sample_rate)
                waveforms[rep, idx] = samples.astype(np.float32, copy=False)

                if progress_every and (idx % progress_every == 0 or idx == n_steps - 1):
                    elapsed = time.perf_counter() - t0
                    print(
                        f"[rep {rep+1}/{n_rep}] [{idx+1}/{n_steps}] "
                        f"{scan_axis}={val:.3f} um  elapsed={elapsed:.2f}s"
                    )

    finally:
        # Save
        np.savez_compressed(
            npz_path,
            positions=positions,
            waveforms=waveforms,
            sample_rate=sample_rate,
            num_samples=num_samples,
            channels=np.array(channel_names),
            ai_physical=ai_physical,
            id=ID,
            created=ts,
            scan_axis=scan_axis,
            scan_values=scan_values,
            n_rep=n_rep,
            axes_mapping=np.array(["x->x_fine", "y->y_fine", "z->z_coarse"])
        )
        meta = dict(
            ID=ID, created=ts,
            n_rep=n_rep, n_steps=n_steps,
            sample_rate=float(sample_rate), num_samples=int(num_samples),
            channels=list(channel_names), ai_physical=ai_physical,
            axes_mapping={"x": "x_fine", "y": "y_fine", "z": "z_coarse"},
            scan_axis=scan_axis,
            scan_values=[float(v) for v in scan_values],
            consts=dict(x=float(const_x), y=float(const_y), z=float(const_z)),
            npz_file=os.path.basename(npz_path)
        )
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        try:
            task.stop()
            task.close()
        except Exception:
            pass

    result = {"save_dir": save_dir, "npz_path": npz_path, "meta_path": meta_path}
    if return_data:
        result.update({"positions": positions, "waveforms": waveforms})

    print(f"Data saved to: {npz_path}")
    return result


def angle_estimation(z_start, z_end, r_start, r_end, stage_position):
    stage_start = stage_position[0]
    stage_end = stage_position[-1]
    stage_position_II = stage_position - stage_start  # um, to make it all positive
    # Estimate the tilt angle for z and r
    Tilt_angle_z = np.arctan((z_end - z_start) / (stage_end - stage_start)) * 180 / np.pi
    Tilt_angle_r = np.arctan((r_end - r_start) / (stage_end - stage_start)) * 180 / np.pi
    
    print(f"Tilt angle in Z-plane: {Tilt_angle_z:.2f} deg")
    print(f"Tilt angle in R-plane: {Tilt_angle_r:.2f} deg")

    # Find the z and r position for each stage position
    z_pos = np.zeros(len(stage_position))
    r_pos = np.zeros(len(stage_position))
    for i in range(len(stage_position)):
        z_pos[i] = z_start + stage_position_II[i] * np.tan(Tilt_angle_z * np.pi / 180)
        r_pos[i] = r_start + stage_position_II[i] * np.tan(Tilt_angle_r * np.pi / 180)
    return z_pos, r_pos

def measure_open(N, duration): # duration is one cycle of the waveform
    # create folder if not exist
    if not os.path.exists(f'data/{foldername}'):
        os.makedirs(f'data/{foldername}')

    # update the filename with stage position
    filename_ = filename + '_' + str(i)

    ctrl = get_controller()

    # Setting the waveform for the open loop measurement
    rate_hz = 1000                          # software update rate (Hz)
    dt = 1.0 / rate_hz
    freq = 1/duration                             # sine frequency (Hz)
    amplitude = 30.0                        # volts peak
    offset = 37.5                           # volts DC bias

    num_cycles = N
    dt = 1.0 / rate_hz
    total_time = num_cycles / freq              # seconds of drive
    total_steps = int(total_time * rate_hz)     # how many voltage updates
    
        
    # DAQ settings
    ai_physical = "cDAQ1Mod1/ai0:3"
    sample_rate = 20e3  # Hz
    buffer_time = 0.1  # s
    meas_time = total_time + buffer_time  # s
    total_samples = int(meas_time * sample_rate)

    ## Create a task
    task = nidaqmx.Task()
    task.ai_channels.add_ai_voltage_chan(ai_physical)
    task.timing.cfg_samp_clk_timing(
        rate=sample_rate,
        sample_mode=AcquisitionType.FINITE,
        samps_per_chan=total_samples
    )

    #### open loop measurement ####
    task.start()

    if 1: # sine wave
        for step in range(total_steps):
            t = step * dt
            
            
            # sine wave
            voltage = offset + amplitude * np.sin(2 * np.pi * freq * t) # sine
            # triangle wave
            voltage = offset + amplitude * (2 * (t * freq % 1) - 1)  # triangle wave adjustment
            ctrl.set_voltage('z_open', voltage)
            time.sleep(dt)

     else: # triangle wave
        for step in range(total_steps):
            t = step * dt
            # triangle wave
            voltage = offset + amplitude * (2 * (t * freq % 1) - 1)  # triangle wave adjustment
            ctrl.set_voltage('z_open', voltage)
            time.sleep(dt)   

    task.wait_until_done(timeout=WAIT_INFINITELY)
    data = task.read(number_of_samples_per_channel=total_samples)
    task.close()

    return OrderedDict([
    ('data', np.array(data)),
    ('sample_rate', sample_rate),
    ('total_samples', total_samples),
    ('meas_time', meas_time),
    ('wavetime', wavetime),
    ('N', N),
    ('wavetable', wavetable),])

def multiple_points_scan(foldername, filename, stage_position, y_pos, r_pos, N=50):
    """
    Multiple point scanning function.
    Axis mapping: x -> x_fine, y -> y_fine, z -> z_coarse
    Note: stage_move() has been removed - implement your own stage control if needed
    """
    for i in range(len(stage_position)):
        # create folder if not exist
        if not os.path.exists(f'data/{foldername}'):
            os.makedirs(f'data/{foldername}')

        # update the filename with stage position
        filename_ = filename + '_' + str(i)

        # Move to initial position
        piezo_move(target_x=XXX, target_y=YYY, target_z=ZZZ)
        
        # Note: stage_move() has been removed
        # If you need to move a stage motor, implement it here
        print(f"Stage position: {stage_position[i]} um (stage_move removed - implement if needed)")
        time.sleep(1)  # wait for the stage to settle

        # move the piezo close to the fiber for scanning
        piezo_move(target_x=r_pos[i], target_y=y_pos[i], target_z=ZZZ)

        # Implement Z-scan until the prominent peak position is found
        peak_prominence = 0
        desired_prominence = 0.1
        x_pos = r_pos[i]

        while peak_prominence < desired_prominence:
            # Scan z axis
            res_0 = measure_piezo_sweep(
                ID=filename_, 
                scan_axis='y',
                scan_values=np.linspace(z_pos[i]-15, z_pos[i] + 15, 50), 
                const_x=x_pos, 
                const_y=YYY
            )
            path = res_0['npz_path']
            # Note: You'll need to implement analyze_gaussian or import it
            res_ana = analyze_gaussian(path)
            peak_prominence = res_ana['popt'][0]
            x_pos = x_pos + 0.2
            
            # Temporary workaround if analyze_gaussian not available
            print(f"Warning: analyze_gaussian not implemented, breaking loop")
            break
        
        # Find the peak position
        new_Y_pos = res_ana['peak_position']

        # move to the new peak position
        piezo_move(target_x=x_pos, target_ynew_Y_pos, target_z=ZZZ)

        # Note: measure_piezo_open needs to be implemented for new hardware
        res = measure_open(N, duration=0.5) # 500 ms scan, N times

        # save
        with open(f'data/{foldername}/{filename_}.npy', 'wb') as f:
            np.save(f, res)
        print(f"Data saved to: data/{foldername}/{filename_}.npy")
