import os
import json
import time
import datetime
from typing import Sequence, Dict, Any, Optional
from collections import OrderedDict
import serial

import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, WAIT_INFINITELY
from pipython import GCSDevice, pitools

def stage_getpos():
    ser = serial.Serial('COM8', 9600)
    command = 'Q:\r\n'
    ser.write(command.encode())
    msg = ser.readline().decode().strip()
    position_str = msg.split(',')[0].strip()
    position = int(position_str.replace(" ", ""))
    ser.close()
    return position

def stage_move(pos):
    ser = serial.Serial('COM8', 9600)
    # set
    direction = '+' if pos >= 0 else '-'
    abs_pos = abs(pos)
    command = f'A:1{direction}P{abs_pos}\r\n'
    ser.write(command.encode())
    ack = ser.readline().decode().strip()
    # move
    ser.write(b'G:\r\n')
    ack = ser.readline().decode().strip()
    # wait
    while True:
        time.sleep(0.1)
        ser.write(b'!:\r\n')
        status = ser.readline().decode().strip()
        if status == 'R':
            break
    # get
    ser.close()
    pos = stage_getpos()
    ser.close()
    return pos

def piezo_getpos(
    pi_serialnum: str = "",
) -> Dict[str, float]:
    """
    PI ピエゾの現在位置を取得する。
    軸マッピング: 1=x, 2=y, 3=z

    Parameters
    ----------
    pi_serialnum : str
        接続する PI デバイスの USB シリアル番号（空なら自動検出）。

    Returns
    -------
    dict: {"x": float, "y": float, "z": float}
        現在位置 [µm]
    """
    pidevice = GCSDevice()
    try:
        #pidevice.InterfaceSetupDlg()
        pidevice.ConnectUSB(serialnum=pi_serialnum)
        pitools.setservo(pidevice, ['1', '2', '3'], [True, True, True])
        pos = pidevice.qPOS()  # {'1': x, '2': y, '3': z}
        return {"x": float(pos['1']), "y": float(pos['2']), "z": float(pos['3'])}
    finally:
        try:
            pidevice.CloseConnection()
        except Exception:
            pass


def piezo_move(
    *,
    target_x: Optional[float] = None,
    target_y: Optional[float] = None,
    target_z: Optional[float] = None,
    pi_serialnum: str = "",
) -> Dict[str, float]:
    """
    PI ピエゾを指定位置 (µm) に移動する。未指定の軸は現在位置を維持。
    軸マッピング: 1=x, 2=y, 3=z

    Parameters
    ----------
    target_x, target_y, target_z : float or None
        目標位置 [µm]。None の軸は現在値を採用。
    pi_serialnum : str
        接続する PI デバイスの USB シリアル番号（空なら自動検出）。

    Returns
    -------
    dict: {"x": float, "y": float, "z": float}
        移動後に qPOS() で取得した実位置 [µm]。
    """
    pidevice = GCSDevice()
    try:
        # 接続 & サーボ有効化
        pidevice.ConnectUSB(serialnum=pi_serialnum)
        pitools.setservo(pidevice, ['1', '2', '3'], [True, True, True])

        # 現在位置を取得
        cur = pidevice.qPOS()  # {'1': x, '2': y, '3': z}
        x = float(cur['1']) if target_x is None else float(target_x)
        y = float(cur['2']) if target_y is None else float(target_y)
        z = float(cur['3']) if target_z is None else float(target_z)

        # 移動
        pitools.moveandwait(pidevice, ['1', '2', '3'], [x, y, z])

        # 実位置を返す
        pos = pidevice.qPOS()
        return {"x": float(pos['1']), "y": float(pos['2']), "z": float(pos['3'])}

    finally:
        try:
            pitools.setservo(pidevice, ['1', '2', '3'], [False, False, False])
            pidevice.CloseConnection()
        except Exception:
            pass


def counter(
    ai_channel: str = "cDAQ1Mod1/ai0",
    sample_rate: float = 10_000,
    num_samples: int = 2_000,
    window_size: int = 100,
    update_interval: float = 0.1,
    title: str = "Real-time Averaged Voltage"  # 未使用（互換性のため残置）
) -> None:
    """
    連続サンプリングでアナログ電圧を取得し、num_samples ごとに平均。
    ライブプロット機能は削除。最新 window_size 個の平均値のみ保持し、
    update_interval ごとに標準出力へ統計情報を出力。Ctrl+C で終了。

    Parameters
    ----------
    ai_channel : str        例: "cDAQ1Mod1/ai0"
    sample_rate : float     サンプリングレート [Hz]
    num_samples : int       平均化に使うサンプル数
    window_size : int       内部バッファに保持する平均値の点数
    update_interval : float ログ出力間隔 [s]
    title : str             （未使用・互換用）
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
        print("手動で停止されました。")

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
    plot_every_points: int = 50,            # 未使用（互換のため残置）
    n_rep: int = 1,
) -> Dict[str, Any]:
    """
    任意の単一軸 (x, y, z) を掃引して raw 波形を取得・保存（npz）。
    - n_rep 回繰り返し測定できる
    - ライブプロットは削除
    """
    axis_map = {"x": "1", "y": "2", "z": "3"}
    if scan_axis not in axis_map:
        raise ValueError("scan_axis must be 'x','y','z'")
    scan_values = np.asarray(scan_values, dtype=float)
    if scan_values.ndim != 1 or len(scan_values) == 0:
        raise ValueError("scan_values must be a 1D sequence with length > 0")

    n_steps = len(scan_values)
    n_rep = int(max(1, n_rep))

    # 保存先
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

    # PI
    pidevice = GCSDevice()
    pidevice.ConnectUSB(serialnum=pi_serialnum)
    pitools.setservo(pidevice, ['1', '2', '3'], [True, True, True])

    # バッファ
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

    # ---- 繰り返しループ ----
    t0 = time.perf_counter()
    try:
        for rep in range(n_rep):
            pos_dict = {"x": float(const_x), "y": float(const_y), "z": float(const_z)}
            pos_dict[scan_axis] = float(scan_values[0])
            pitools.moveandwait(
                pidevice, ['1', '2', '3'],
                [pos_dict["x"], pos_dict["y"], pos_dict["z"]]
            )

            for idx, val in enumerate(scan_values):
                pos_dict[scan_axis] = float(val)
                pitools.moveandwait(
                    pidevice, ['1', '2', '3'],
                    [pos_dict["x"], pos_dict["y"], pos_dict["z"]]
                )
                if settle_sec > 0:
                    time.sleep(settle_sec)

                pos = pidevice.qPOS()
                x_um = float(pos['1']); y_um = float(pos['2']); z_um = float(pos['3'])
                positions[rep, idx] = (x_um, y_um, z_um)

                samples = _acquire_block(task, num_samples, sample_rate)
                waveforms[rep, idx] = samples.astype(np.float32, copy=False)

                if progress_every and (idx % progress_every == 0 or idx == n_steps - 1):
                    elapsed = time.perf_counter() - t0
                    print(
                        f"[rep {rep+1}/{n_rep}] [{idx+1}/{n_steps}] "
                        f"{scan_axis}={val:.3f} um  elapsed={elapsed:.2f}s"
                    )

    finally:
        # 保存
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
            axes_mapping=np.array(["1->x", "2->y", "3->z"])
        )
        meta = dict(
            ID=ID, created=ts,
            n_rep=n_rep, n_steps=n_steps,
            sample_rate=float(sample_rate), num_samples=int(num_samples),
            channels=list(channel_names), ai_physical=ai_physical,
            axes_mapping={"1": "x", "2": "y", "3": "z"},
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
        try:
            pitools.setservo(pidevice, ['1', '2', '3'], [False, False, False])
            pidevice.CloseConnection()
        except Exception:
            pass

    result = {"save_dir": save_dir, "npz_path": npz_path, "meta_path": meta_path}
    if return_data:
        result.update({"positions": positions, "waveforms": waveforms})

    print(f"Data saved to: {npz_path}")
    return result

def acquire_V():
    # Just obtain optical signal with DAQ and return current voltage
    
    # DAQ 
    ai_physical = "cDAQ1Mod1/ai0"
    sample_rate = 20e3  # Hz
    meas_time = 0.05  # s
    total_samples = int(meas_time * sample_rate)

    ## Create a task
    task = nidaqmx.Task()
    task.ai_channels.add_ai_voltage_chan(ai_physical)
    task.timing.cfg_samp_clk_timing(
        rate=sample_rate,
        sample_mode=AcquisitionType.FINITE,
        samps_per_chan=total_samples
    )

    # Start measurement
    task.start()
    task.wait_until_done(timeout=WAIT_INFINITELY)
    data = task.read(number_of_samples_per_channel=total_samples)
    task.close()

    V = np.mean(data)
    return V

    def Z_scan_half(x, y, list):
        V_tail = 0.001
        # list is scanning range of z axis
        V_list = []
        piezo_move(target_x=x, target_y=y, target_z=list[0])
        V_list.append(acquire_V())
        for i in range(1,len(list)):
            piezo_move(target_x=x, target_y=y, target_z=list[i])
            V_list.append(acquire_V())
            if abs(V_list[i] - V_list[i-1]) > V_tail:
                if V_list[i] - V_list[i-1] < 0:
                    break 
        
        z_pos = list[i]
        return z_pos



def measure_piezo_open(wavetable: int, N: int):
    # Piezo
    ## piezo connect
    pidevice = GCSDevice()
    pidevice.ConnectUSB(serialnum="")
    pidevice.SVO(['1', '2', '3'], [False, False, False])

    ## Waveform は外部で定義済みを想定（GUI 等）
    ## 一旦 WG 停止
    pidevice.send("WGO 2 0")

    ## 波形テーブルをチャンネルに割当
    wavepoints = pidevice.qWAV(wavetable, 1)[wavetable][1]
    wavetime = wavepoints / 20000  # s（例：20 kS/s 基準）
    pidevice.send(f'WSL 2 {wavetable}')

    ## 繰り返し回数
    pidevice.send(f'WGC 2 {N}')

    ## Trigger 設定
    TrigOut = 1  # digital output 1
    CTOPam1, Value1 = 2, 2  # Axis set to 2(y)
    CTOPam2, Value2 = 3, 9  # Mode: Generator Pulse Trigger
    CTOPam3, Value3 = 7, 1  # Polarity: High when active
    pidevice.send(f'CTO {TrigOut} {CTOPam1} {Value1}')
    pidevice.send(f'CTO {TrigOut} {CTOPam2} {Value2}')
    pidevice.send(f'CTO {TrigOut} {CTOPam3} {Value3}')

    ## Enable trigger
    PointNumber = 1  # at which emit 'High' Amp
    Switch = 1      # rising edge
    # pidevice.send(f'TWS {TrigOut} {PointNumber} {Switch}')

    # DAQ
    ai_physical = "cDAQ1Mod1/ai0:3"
    sample_rate = 20e3  # Hz
    buffer_time = 0.1  # s
    meas_time = wavetime * N + buffer_time  # s
    total_samples = int(meas_time * sample_rate)

    ## Create a task
    task = nidaqmx.Task()
    task.ai_channels.add_ai_voltage_chan(ai_physical)
    task.timing.cfg_samp_clk_timing(
        rate=sample_rate,
        sample_mode=AcquisitionType.FINITE,
        samps_per_chan=total_samples
    )

    # Start measurement
    pidevice.send("SVA 2 30")
    time.sleep(0.1)
    pidevice.send("WGO 2 0")
    pidevice.send(f'TWS {TrigOut} {PointNumber} {Switch}')
    task.start()
    pidevice.send("WGO 2 1")
    task.wait_until_done(timeout=WAIT_INFINITELY)
    data = task.read(number_of_samples_per_channel=total_samples)
    task.close()
    pidevice.send("WGO 2 0")
    pidevice.CloseConnection()

    return OrderedDict([
        ('data', np.array(data)),
        ('sample_rate', sample_rate),
        ('total_samples', total_samples),
        ('meas_time', meas_time),
        ('wavetime', wavetime),
        ('N', N),
        ('wavetable', wavetable),
    ])

def angle_estimation(z_start, z_end, r_start, r_end, stage_position):
    stage_start = stage_position[0]
    stage_end = stage_position[-1]
    stage_position_II = stage_position - stage_start  # um, to make it all positive
    # Estimate the tilt angle for z and r
    Tilt_angle_z = np.arctan((z_end - z_start) / (100)) * 180 / np.pi
    Tilt_angle_r = np.arctan((r_end - r_start) / (100)) * 180 / np.pi
    
    print(f"Tilt angle in Z-plane: {Tilt_angle_z:.2f} deg")
    print(f"Tilt angle in R-plane: {Tilt_angle_r:.2f} deg")

    # Find the z and r position for each stage position
    z_pos = np.zeros(len(stage_position))
    r_pos = np.zeros(len(stage_position))
    for i in range(len(stage_position)):
        z_pos[i] = z_start + stage_position_II[i] * np.tan(Tilt_angle_z * np.pi / 180)
        r_pos[i] = r_start + stage_position_II[i] * np.tan(Tilt_angle_r * np.pi / 180)
    return z_pos, r_pos

def multiple_points_scan(foldername, filename,stage_position, z_pos, r_pos,N = 50):
    for i in range(len(stage_position)):
        # create folder if not exist
        if not os.path.exists(f'data/{foldername}'):
            os.makedirs(f'data/{foldername}')

        # update the filename with stage position
        filename_ = filename + '_' + str(i)

        # move the motor stage
        piezo_move(target_x=50, target_y=70, target_z=50)
        stage_move(stage_position[i])
        time.sleep(1) # wait for the stage to settle
        print(f"Stage moved to: {stage_position[i]} um")

        # move the piezo close to the fiber for scanning
        piezo_move(target_x=r_pos[i], target_y=70, target_z=z_pos[i])

        # Implement Z-scan until the prominent peak position is found
        peak_prominence = 0
        desired_prominence = 0.1
        x_pos = r_pos[i]

        while peak_prominence < desired_prominence:
            res_0 = measure_piezo_sweep(ID=filename_, scan_axis='z', scan_values=np.linspace(z_pos[i]-15, z_pos[i] + 15, 50), const_x=x_pos, const_y=70)
            path = res_0['npz_path']
            res_ana = analyze_gaussian(path)
            peak_prominence = res_ana['popt'][0]
            x_pos = x_pos + 0.2
        
        # Find the peak position
        new_Z_pos = res_ana['peak_position']

        # move to the new peak position
        piezo_move(target_x=x_pos, target_y=70, target_z=new_Z_pos)

        # start scanning
        res = measure_piezo_open(1, N) # 500 ms scan, N times

        # save
        with open(f'data/{foldername}/{filename_}.npy', 'wb') as f:
            np.save(f, res)
        print(f"Data saved to: data/{foldername}/{filename_}.npy")





## Archive ##################################################################
# def enable_output():
#     # Open serial port (check COM port in Device Manager)
#     ser = serial.Serial(
#         port='COM9',      
#         baudrate=115200,
#         bytesize=8,
#         parity='N',
#         stopbits=1,
#         timeout=1
#     )
#     # Enable HV output
#     ser.write(b'enable=1\r')
#     time.sleep(0.5)

#     # Query current state
#     ser.write(b'enable?\r')
#     print(ser.readline().decode().strip())
    
#     ser.close()

# def disable_output():
#     # Open serial port (check COM port in Device Manager)
#     ser = serial.Serial(
#         port='COM9',      
#         baudrate=115200,
#         bytesize=8,
#         parity='N',
#         stopbits=1,
#         timeout=1
#     )

#     # Disable HV output (safety first)
#     ser.write(b'enable=0\r')
#     time.sleep(0.5)

#     # Query current state
#     ser.write(b'enable?\r')
#     print(ser.readline().decode().strip())
    
#     ser.close()

# def multiple_points_scan(foldername, filename,stage_position, z_pos, r_pos,N = 50, freq = 20):
#     wavetime = 1 / freq  # s
#     for i in range(len(stage_position)):
#         # create folder if not exist
#         if not os.path.exists(f'data/{foldername}'):
#             os.makedirs(f'data/{foldername}')

#         # update the filename with stage position
#         filename_ = filename + '_' + str(stage_position[i])

#         # move the motor stage
#         stage_move(stage_position[i])
#         time.sleep(0.5) # wait for the stage to settle
#         print(f"Stage moved to: {stage_position[i]} um")

#         # move the piezo close to the fiber for scanning
#         piezo_move(target_x=r_pos[i], target_y=70, target_z=z_pos[i])

#         # DAQ setting
#         ai_physical = "cDAQ1Mod1/ai0:3"
#         sample_rate = 20e3  # Hz
#         buffer_time = 0.1  # s
#         meas_time = wavetime * N + buffer_time  # s
#         total_samples = int(meas_time * sample_rate)

#         ## Create a task
#         task = nidaqmx.Task()
#         task.ai_channels.add_ai_voltage_chan(ai_physical)
#         task.timing.cfg_samp_clk_timing(
#             rate=sample_rate,
#             sample_mode=AcquisitionType.FINITE,
#             samps_per_chan=total_samples
#             )
        

#         # start scanning
#         task.start()
#         enable_output()
#         task.wait_until_done(timeout=WAIT_INFINITELY)
#         disable_output()
#         data = task.read(number_of_samples_per_channel=total_samples)
#         task.close()    

#         # save
#         with open(f'data/{foldername}/{filename_}.npy', 'wb') as f:
#             np.save(f, res)
#         print(f"Data saved to: data/{foldername}/{filename_}.npy")


