import os, csv, json    
from typing import Optional, Sequence, Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.optimize import curve_fit
from dataclasses import dataclass
import pandas as pd
from matplotlib.colors import to_rgba
from collections import OrderedDict
from dataclasses import dataclass
from eeequation import *
from beta_vs_radius_plot import CompactFiberSolver

COLORS = {
    "probe": "tab:blue",
    "transmission": "tab:orange",
    "interferometer": "tab:green",
    "feedback": "tab:red",
}
DISPLAY = {
    "probe": "Probe",
    "transmission": "Transmission",
    "interferometer": "Interferometer",
    "feedback": "Feedback signal",
}
ORDER = ["probe", "transmission", "interferometer", "feedback"]
FIT_CHANNELS = ("probe", "interferometer")
PITCH_BOX_ALPHA = 0.6


def _iter_reps_from_npz(npz_obj):
    wave = np.asarray(npz_obj["waveforms"])
    pos  = np.asarray(npz_obj["positions"])
    if wave.ndim == 4:
        for r in range(wave.shape[0]):
            yield r, pos[r], wave[r]
    elif wave.ndim == 3:
        yield 0, pos, wave
    else:
        raise ValueError(f"Unexpected waveforms.ndim={wave.ndim}")

def analyze_gaussian(
    npz_path: str,
    *,
    scan_axis: Optional[str] = None,
    ddof: int = 1,
    save_png: bool = True,
    show: bool = False,
    dpi: int = 200,
    probe_ylim: Optional[Tuple[float, float]] = None,
    channel_names: Optional[Sequence[str]] = None,
    plot_channels: Optional[Sequence[str]] = None,  # ★追加
) -> Dict[str, Any]:
    npz = np.load(npz_path)
    wave_any = np.asarray(npz["waveforms"])
    channels  = list(npz.get("channels", [f"ch{i}" for i in range(wave_any.shape[-2])]))
    if channel_names is not None:
        channels = list(channel_names)
    npz_axis  = str(npz.get("scan_axis", "x"))
    npz_id    = str(npz.get("id", os.path.basename(os.path.dirname(npz_path)) or "ID"))
    scan_vals_all = np.asarray(npz.get("scan_values", None))
    axis = (scan_axis or npz_axis).lower()
    idx_xyz = {"x":0, "y":1, "z":2}
    if axis not in idx_xyz:
        raise ValueError(f"scan_axis must be x/y/z, got: {axis}")

    # チャンネル解決
    def _find(name_substr, default):
        for i, nm in enumerate(channels):
            if name_substr in nm.lower(): return i
        return default
    idx_probe  = _find("probe", 0)
    idx_trans  = _find("trans", 1 if len(channels)>1 else 0)
    idx_interf = _find("interf", 2 if len(channels)>2 else 0)
    idx_feed   = _find("feed",  3 if len(channels)>3 else 0)

    all_idx  = {"probe": idx_probe, "transmission": idx_trans,
                "interferometer": idx_interf, "feedback": idx_feed}
    all_keys = list(all_idx.keys())

    # プロットするチャンネルを決定
    if plot_channels is None:
        plot_channels = all_keys  # 全部
    else:
        # 小文字化して受け付ける
        plot_channels = [ch.lower() for ch in plot_channels if ch.lower() in all_idx]

    per_rep = []
    for rep_idx, positions, waveforms in _iter_reps_from_npz(npz):
        M, C, N = waveforms.shape
        if scan_vals_all is not None and scan_vals_all.shape[0] == positions.shape[0]:
            x = scan_vals_all.astype(float)
        else:
            x = positions[:, idx_xyz[axis]].astype(float)

        means = waveforms.mean(axis=2, dtype=np.float64)
        stds  = waveforms.std(axis=2, ddof=ddof, dtype=np.float64)

        y_probe = means[:, all_idx["probe"]]

        def gauss_off(z, A, z0, sigma, C0):
            return A * np.exp(- (z - z0)**2 / (2.0 * sigma**2)) + C0

        span   = max(1e-12, float(np.max(x) - np.min(x)))
        A0     = float(np.max(y_probe) - np.median(y_probe))
        z0_0   = float(x[int(np.argmax(y_probe))])
        sigma0 = 0.15 * span
        C0     = float(np.median(y_probe))
        p0 = [A0 if np.isfinite(A0) else 1.0,
              z0_0 if np.isfinite(z0_0) else float(np.mean(x)),
              sigma0 if np.isfinite(sigma0) and sigma0 > 0 else span*0.1,
              C0 if np.isfinite(C0) else 0.0]
        lb = [-np.inf, float(np.min(x))-span, 1e-12, -np.inf]
        ub = [ np.inf, float(np.max(x))+span,  np.inf,  np.inf]

        popt, pcov = curve_fit(gauss_off, x, y_probe, p0=p0, bounds=(lb, ub), maxfev=60000)
        A_fit, z0_fit, sigma_fit, C_fit = map(float, popt)

        # ★ 指定チャンネルだけ描画
        fig, axes = plt.subplots(len(plot_channels), 1, sharex=True, figsize=(6.4, 3.0*len(plot_channels)))
        if len(plot_channels) == 1:
            axes = [axes]

        for ax, key in zip(axes, plot_channels):
            ch_idx = all_idx[key]
            col = COLORS[key]; ecol = to_rgba(col, 0.3)
            y = means[:, ch_idx]; e = stds[:, ch_idx]
            ax.errorbar(x, y, yerr=e, fmt='none', ecolor=ecol, elinewidth=1.0, capsize=2)
            ax.plot(x, y, 'o', mfc=col, mec=col, ms=4)
            if key == "probe":
                xf = np.linspace(np.min(x), np.max(x), 1200)
                ax.plot(xf, gauss_off(xf, *popt), '-', color="black", lw=1.6)
                note = f"peak = {z0_fit:.1f} µm"
                ax.text(0.02, 0.05, note, transform=ax.transAxes, ha='left', va='bottom',
                        fontsize=9, bbox=dict(facecolor='white', edgecolor='none', alpha=PITCH_BOX_ALPHA))
                if probe_ylim is not None:
                    ax.set_ylim(*probe_ylim)
            ax.set_ylabel(f"{DISPLAY[key]} (V)")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel(f"{axis} (µm)")
        fig.suptitle(f"{npz_id} (rep {rep_idx})")
        fig.tight_layout(rect=[0,0,1,0.97])

        png_path = None
        if save_png:
            base = os.path.splitext(npz_path)[0]
            png_path = f"{base}_rep{rep_idx:02d}_gauss_probe.png"
            fig.savefig(png_path, dpi=dpi)
        if show:
            plt.show()
        plt.close(fig)

        per_rep.append(dict(
            rep=rep_idx,
            axis=axis, x=x,
            means=means, stds=stds,
            popt=np.array([A_fit, z0_fit, sigma_fit, C_fit]),
            pcov=pcov,
            peak_position=z0_fit,
            png_path=png_path,
            channels=channels,
            id=npz_id,
        ))

    head = per_rep[0]
    return dict(
        per_rep=per_rep,
        **{k: head.get(k) for k in (
            "axis","x","means","stds","popt","pcov","peak_position","png_path","channels","id")}
    )

from typing import Optional, Tuple, Sequence, Dict, Any
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def analyze_exp(
    npz_path: str,
    *,
    x_range: Optional[Tuple[float, float]] = None,   # フィット用の x 範囲（プロットは全点）
    xref: Optional[float] = None,                    # 指数の基準点（省略可）
    scan_axis: Optional[str] = None,
    ddof: int = 1,
    save_png: bool = True,
    show: bool = False,
    dpi: int = 200,
    probe_ylim: Optional[Tuple[float, float]] = None,
    channel_names: Optional[Sequence[str]] = None,
    plot_channels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    y(x) = A * exp(B * (x - x_ref)) + C をフィット。
    - フィットは x_range 内の点のみ。
    - プロットは範囲外の点も表示（範囲内:通常、範囲外:薄色）。
    - 係数は [A, B, C, x_ref] を popt に格納。
    """
    npz = np.load(npz_path, allow_pickle=False)
    wave_any = np.asarray(npz["waveforms"])
    channels = list(npz.get("channels", [f"ch{i}" for i in range(wave_any.shape[-2])]))
    if channel_names is not None:
        channels = list(channel_names)
    npz_axis = str(npz.get("scan_axis", "x"))
    npz_id   = str(npz.get("id", os.path.basename(os.path.dirname(npz_path)) or "ID"))
    scan_vals_all = np.asarray(npz.get("scan_values", None))
    axis = (scan_axis or npz_axis).lower()
    idx_xyz = {"x": 0, "y": 1, "z": 2}
    if axis not in idx_xyz:
        raise ValueError(f"scan_axis must be x/y/z, got: {axis}")

    # チャンネル解決
    def _find(name_substr, default):
        for i, nm in enumerate(channels):
            if name_substr in nm.lower():
                return i
        return default
    idx_probe  = _find("probe", 0)
    idx_trans  = _find("trans", 1 if len(channels) > 1 else 0)
    idx_interf = _find("interf", 2 if len(channels) > 2 else 0)
    idx_feed   = _find("feed",  3 if len(channels) > 3 else 0)

    all_idx  = {"probe": idx_probe, "transmission": idx_trans,
                "interferometer": idx_interf, "feedback": idx_feed}
    all_keys = list(all_idx.keys())

    # プロットするチャンネル
    if plot_channels is None:
        plot_channels = all_keys
    else:
        plot_channels = [ch.lower() for ch in plot_channels if ch.lower() in all_idx]

    def exp_off(x, A, B, C, x_ref):
        return A * np.exp(B * (x - x_ref)) + C

    per_rep = []
    for rep_idx, positions, waveforms in _iter_reps_from_npz(npz):
        M, Cn, N = waveforms.shape
        if scan_vals_all is not None and scan_vals_all.shape[0] == positions.shape[0]:
            x_full = scan_vals_all.astype(float)
        else:
            x_full = positions[:, idx_xyz[axis]].astype(float)

        means = waveforms.mean(axis=2, dtype=np.float64)
        stds  = waveforms.std(axis=2, ddof=ddof, dtype=np.float64)

        # フィット範囲マスク
        if x_range is not None:
            xmin, xmax = float(x_range[0]), float(x_range[1])
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            mask = (x_full >= xmin) & (x_full <= xmax)
        else:
            mask = np.ones_like(x_full, dtype=bool)

        if mask.sum() < 4:
            raise ValueError("指定された x_range 内のデータ点が少なすぎます（>=4 点必要）")

        # フィット用データ
        x_in = x_full[mask]
        y_probe_in = means[:, all_idx["probe"]][mask]

        # 初期値推定
        q20 = np.quantile(y_probe_in, 0.2)
        C0  = float(q20)
        idx_max = int(np.argmax(np.abs(y_probe_in - C0)))
        A0 = float(y_probe_in[idx_max] - C0)
        if np.isclose(A0, 0.0):
            A0 = float(y_probe_in.max() - C0 + 1e-6)
        x_ref = float(x_in.min()) if xref is None else float(xref)

        eps = max(1e-12, 1e-6*np.ptp(y_probe_in))
        y_adj = y_probe_in - C0
        ok = y_adj > eps
        if ok.sum() >= 2:
            X = x_in[ok] - x_ref
            Y = np.log(y_adj[ok])
            denom = float(np.sum((X - X.mean())**2))
            B0 = float(np.sum((X - X.mean())*(Y - Y.mean())) / denom) if denom > 0 else 1.0
        else:
            B0 = 1.0

        p0 = [A0 if np.isfinite(A0) else 1.0,
              B0 if np.isfinite(B0) else 1.0,
              C0 if np.isfinite(C0) else 0.0]
        lb = [-np.inf, -1e3, -np.inf]
        ub = [ np.inf,  1e3,  np.inf]

        def _model(xv, A, B, C):
            return exp_off(xv, A, B, C, x_ref)

        popt, pcov = curve_fit(_model, x_in, y_probe_in, p0=p0, bounds=(lb, ub), maxfev=60000)
        A_fit, B_fit, C_fit = map(float, popt)

        # ===== プロット：全点を表示、フィット曲線はマスク内だけ =====
        fig, axes = plt.subplots(len(plot_channels), 1, sharex=True,
                                 figsize=(6.4, 3.0*len(plot_channels)))
        if len(plot_channels) == 1:
            axes = [axes]

        # フィット曲線はマスク範囲のみに限定
        xf = np.linspace(x_in.min(), x_in.max(), 1200)
        y_fit = exp_off(xf, A_fit, B_fit, C_fit, x_ref)

        for ax, key in zip(axes, plot_channels):
            ch_idx = all_idx[key]
            col = COLORS[key]
            ecol_in  = to_rgba(col, 0.35)
            ecol_out = to_rgba(col, 0.12)

            y_all = means[:, ch_idx]
            e_all = stds[:,  ch_idx]

            # 範囲外（薄色）
            x_out = x_full[~mask]
            if x_out.size:
                y_out = y_all[~mask]; e_out = e_all[~mask]
                ax.errorbar(x_out, y_out, yerr=e_out, fmt='none', ecolor=ecol_out,
                            elinewidth=1.0, capsize=2)
                ax.plot(x_out, y_out, 'o', mfc=to_rgba(col, 0.35), mec=to_rgba(col, 0.35), ms=3)

            # 範囲内（通常）
            ax.errorbar(x_in, y_all[mask], yerr=e_all[mask], fmt='none', ecolor=ecol_in,
                        elinewidth=1.2, capsize=2)
            ax.plot(x_in, y_all[mask], 'o', mfc=col, mec=col, ms=4)

            if key == "probe":
                # フィット曲線はマスク範囲のみ
                ax.plot(xf, y_fit, '-', color="black", lw=1.6)
                note = (f"A={A_fit:.3g}, B={B_fit:.3g}, C={C_fit:.3g}\n"
                        f"x_ref={x_ref:.3f}, fit:[{x_in.min():.3f},{x_in.max():.3f}]")
                ax.text(0.02, 0.05, note, transform=ax.transAxes, ha='left', va='bottom',
                        fontsize=9, bbox=dict(facecolor='white', edgecolor='none', alpha=PITCH_BOX_ALPHA))
                if probe_ylim is not None:
                    ax.set_ylim(*probe_ylim)

            ax.set_ylabel(f"{DISPLAY[key]} (V)")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel(f"{axis} (µm)")
        title = f"{npz_id} (rep {rep_idx})  exp fit on [{x_in.min():.3f}, {x_in.max():.3f}]"
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        png_path = None
        if save_png:
            base = os.path.splitext(npz_path)[0]
            png_path = f"{base}_rep{rep_idx:02d}_exp_probe.png"
            fig.savefig(png_path, dpi=dpi)
        if show:
            plt.show()
        plt.close(fig)

        per_rep.append(dict(
            rep=rep_idx,
            axis=axis,
            x=x_full,                    # 全 x（従来と同様）
            fit_mask=mask,               # どの点を使ったか
            means=means, stds=stds,
            popt=np.array([A_fit, B_fit, C_fit, float(x_ref)]),
            pcov=pcov,
            png_path=png_path,
            channels=channels,
            id=npz_id,
        ))

    head = per_rep[0]
    return dict(
        per_rep=per_rep,
        **{k: head.get(k) for k in (
            "axis","x","means","stds","popt","pcov","png_path","channels","id")}
    )



# ---------- ユーティリティ ----------
def _load_data(path):
    return OrderedDict(np.load(path, allow_pickle=True).item())

def _sine_model(t, A, f, phi, C):
    """A*sin(2π f t + phi) + C"""
    return A * np.sin(2*np.pi*f*t + phi) + C

def _r2_score(y, y_fit):
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - (ss_res / ss_tot)

def _reduced_chi2(y, y_fit, num_params):
    residuals = y - y_fit
    ss_res = np.sum(residuals ** 2)
    dof = len(y) - num_params
    if dof <= 0:
        return np.nan
    return ss_res / dof

@dataclass
class ROIResult:
    index: int
    roi_start: float   # [nm]  各ROI内の先頭 → 0
    roi_end: float     # [nm]  各ROIの長さ（nm）
    period_ch0: float  # [s]
    period_ch2: float  # [s]
    pitch_nm: float
    pitch_err_nm: float  # 1σ
    r2_ch0: float
    r2_ch2: float

def analyze_snom_mi(
    filename: str,
    period_ms: float,
    start_time: float,
    end_time: float,
    save_prefix: str = "1000ms_fig",
    wavelength_nm: float = 1388.25,
    save_csv: str | None = "results.csv",
    show: bool = True,
    n_rois: int | None = None,
    do_fft: bool = True  # ← 新規追加：FFTを計算＆出力するかどうか
):
    res = _load_data(filename)
    sample_rate = float(res['sample_rate']) # Hz
    dt = 1.0 / sample_rate # s
    N = int(res['total_samples'])
    time = np.linspace(0, dt*(N-1), N) # s
    data = np.asarray(res['data'])

    CH0 = data[0]  # SNOM
    CH2 = data[2]  # MI

    shift = float(period_ms) / 1000.0 # s
    roi_start = float(start_time) # s
    roi_end = float(end_time) # s

    base = os.path.splitext(os.path.basename(filename))[0]
    out_dir = os.path.join(os.path.dirname(filename), base)
    os.makedirs(out_dir, exist_ok=True)

    save_prefix = os.path.join(out_dir, save_prefix)
    if save_csv is not None:
        save_csv = os.path.join(out_dir, save_csv)

    results: list[ROIResult] = []
    i = 0
    while i < n_rois:
        mask = (time > roi_start) & (time < roi_end)
        if np.count_nonzero(mask) < 8:
            break

        t_roi = time[mask]
        ch0_roi = CH0[mask]
        ch2_roi = CH2[mask]

        # --- fitting ---
        FS_length = 30_000 # nm
        k = FS_length / (wavelength_nm / 2.0)  #
        T0 = period_ms / 2 / 1000 / k
        r20_best = -np.inf
        r22_best = -np.inf
        best_popt0 = None
        best_pcov0 = None
        best_popt2 = None
        best_pcov2 = None
        for r in np.linspace(0.7, 1.3, 20):
            T = T0 * r
            f = 1.0 / T
            param0 = [(max(ch0_roi)-min(ch0_roi))/2, f, 0.0, (max(ch0_roi)+min(ch0_roi))/2]
            param2 = [(max(ch2_roi)-min(ch2_roi))/2, f, 0.0, (max(ch2_roi)+min(ch2_roi))/2]

            try:
                popt0, pcov0 = curve_fit(_sine_model, t_roi, ch0_roi, p0=param0)
                r2 = _r2_score(ch0_roi, _sine_model(t_roi, *popt0))
                if r2 > r20_best:
                    r20_best = r2
                    best_popt0 = popt0
                    best_pcov0 = pcov0
            except Exception as e:
                pass
            try:
                popt2, pcov2 = curve_fit(_sine_model, t_roi, ch2_roi, p0=param2, bounds=(0, np.inf))
                r2 = _r2_score(ch2_roi, _sine_model(t_roi, *popt2))
                if r2 > r22_best:
                    r22_best = r2
                    best_popt2 = popt2
                    best_pcov2 = pcov2
            except Exception as e:
                pass
        
        fit0 = np.zeros_like(t_roi) if best_popt0 is None else _sine_model(t_roi, *best_popt0)
        fit2 = _sine_model(t_roi, *best_popt2)

        resid0 = ch0_roi - fit0
        resid2 = ch2_roi - fit2

        # Period
        f0 = best_popt0[1] if not best_popt0 is None else np.nan
        f2 = best_popt2[1]
        period0 = 1.0 / f0 # s
        period2 = 1.0 / f2 # s

        # s to nm
        scale_nm_per_s = (wavelength_nm / 2) / period2 # nm/s
        x_roi = (t_roi - roi_start) * scale_nm_per_s  # nm

        # r2
        r2_ch0 = _r2_score(ch0_roi, fit0)
        r2_ch2 = _r2_score(ch2_roi, fit2)

        # plotting
        nrows = 4
        fig, axes = plt.subplots(nrows, 1, figsize=(8, 12))

        # SNOM
        axes[0].plot(x_roi, ch0_roi, '.', color="tab:blue")
        axes[0].plot(x_roi, fit0, color="black")
        axes[0].set_ylabel("SNOM probe signal(V)")
        text_snom = (
            "Pitch   = {pitch:.2f} ± {pitch_err:.2f} nm (σ)\n"
            f"R$^2$  = {r2_ch0:.3f}"
        )

        # SNOM residual
        axes[1].plot(x_roi, resid0, '.', color="tab:blue")
        axes[1].axhline(0, lw=1, color="black")
        axes[1].set_ylabel("Residual")

        # MI
        axes[2].plot(x_roi, ch2_roi, '.', color="tab:green")
        axes[2].plot(x_roi, fit2, color="black")
        axes[2].set_ylabel("MI signal (V)")
        axes[2].text(
            0.02, 0.04,
            f"R$^2$  = {r2_ch2:.3f}",
            transform=axes[2].transAxes,
            fontsize=8, va='bottom', ha='left',
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
        )

        # MI residual
        axes[3].plot(x_roi, resid2, '.', color="tab:green")
        axes[3].axhline(0, lw=1, color="black")
        axes[3].set_xlabel("Displacement (nm)")
        axes[3].set_ylabel("Residual")

        # pitch
        pitch = period0 * scale_nm_per_s # nm
        f_err = np.sqrt(best_pcov0[1,1]) if not best_pcov0 is None else np.nan
        period_err = f_err / (f0**2) if not best_pcov0 is None else np.nan
        pitch_err = period_err * scale_nm_per_s if not best_pcov0 is None else np.nan

        axes[0].text(
            0.02, 0.04,
            text_snom.format(pitch=pitch, pitch_err=pitch_err),
            transform=axes[0].transAxes,
            fontsize=8, va='bottom', ha='left',
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
        )

        plt.tight_layout()
        out_png = f"{save_prefix}_{i:02d}.png"
        plt.savefig(out_png, dpi=300)
        if show:
            plt.show()
        else:
            plt.close(fig)

        # roi
        roi_len_s = (roi_end - roi_start)
        roi_start_nm = 0.0
        roi_end_nm = roi_len_s * scale_nm_per_s

        results.append(ROIResult(
            index=i,
            roi_start=roi_start_nm,
            roi_end=roi_end_nm,
            period_ch0=period0,
            period_ch2=period2,
            pitch_nm=pitch,
            pitch_err_nm=pitch_err,
            r2_ch0=r2_ch0,
            r2_ch2=r2_ch2
        ))

        roi_start += shift
        roi_end   += shift
        print(i)
        i += 1

    # csv save
    if save_csv is not None and len(results) > 0:
        import csv
        with open(save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "index",
                "roi_start_nm", "roi_end_nm",
                "period_ch0_s", "period_ch2_s",
                "pitch_nm", "pitch_err_nm",
                "r2_ch0", "r2_ch2"
            ])
            for r in results:
                writer.writerow([
                    r.index,
                    r.roi_start, r.roi_end,
                    r.period_ch0, r.period_ch2,
                    r.pitch_nm, r.pitch_err_nm,
                    r.r2_ch0, r.r2_ch2
                ])

    return results

def plot_pitch_from_csv(csv_files, summary_csv="pitch_summary.csv",
                        save_plot_errorbar=None, save_plot_hist=None,
                        r2_threshold=None, label=None, diameter=False):
    if r2_threshold is None:
        r2_threshold = 0
    if label is not None:
        labels_override = list(label)
    else:
        labels_override = None

    BIN_WIDTH_NM = 0.2
    records = []
    per_file_pitch = []
    labels = []
    all_pitch_for_bins = []

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    if not color_cycle:
        color_cycle = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                       'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                       'tab:olive', 'tab:cyan']

    for idx, fname in enumerate(csv_files):
        df = pd.read_csv(fname)
        default_label = os.path.splitext(os.path.basename(fname))[0]
        label_i = labels_override[idx] if labels_override is not None else default_label

        r2mask = (df["r2_ch0"].to_numpy(dtype=float) >= r2_threshold) & \
                 (df["r2_ch2"].to_numpy(dtype=float) >= r2_threshold)

        n_total = int(len(df))
        df_filt = df[r2mask].copy()
        n_pass  = int(len(df_filt))

        pitch = df_filt["pitch_nm"].to_numpy(dtype=float)
        sigma = df_filt["pitch_err_nm"].to_numpy(dtype=float)

        mask_pitch = np.isfinite(pitch)
        mask_sigma = np.isfinite(sigma) & (sigma > 0)
        mask_weightable = mask_pitch & mask_sigma
        mask_any_finite = mask_pitch

        p_weight = pitch[mask_weightable]
        s_weight = sigma[mask_weightable]
        w = 1.0 / (s_weight ** 2)

        sumw = np.sum(w)
        wmean = float(np.sum(w * p_weight) / sumw)
        sem_w = float(np.sqrt(1.0 / sumw))
        method = "weighted (sigma=pitch_err_nm)"
        n_weighted = int(p_weight.size)

        p_u_all = pitch[mask_any_finite]
        mean_u = float(np.mean(p_u_all)) if p_u_all.size > 0 else np.nan
        std_u  = float(np.std(p_u_all, ddof=1)) if p_u_all.size > 1 else np.nan

        records.append({
            "file": label_i,
            "r2_threshold": r2_threshold,
            "bin_width_nm": BIN_WIDTH_NM,
            "n_points_total": n_total,
            "n_points_pass_r2": n_pass,
            "n_points_weighted": n_weighted,
            "mean_pitch_nm_unweighted": mean_u,
            "std_pitch_nm_unweighted": std_u,
            "mean_pitch_nm_weighted": wmean,
            "sem_pitch_nm_weighted": sem_w,
            "weighting_method": method
        })

        per_file_pitch.append(p_u_all)
        labels.append(label_i)
        if p_u_all.size > 0:
            all_pitch_for_bins.append(p_u_all)

    # save csv
    summary_df = pd.DataFrame(records)
    summary_df.to_csv(summary_csv, index=False)

    # err bar
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary_df))
    means_p = summary_df["mean_pitch_nm_weighted"].to_numpy(dtype=float)
    sems_p  = summary_df["sem_pitch_nm_weighted"].to_numpy(dtype=float)

    # left axis (pitch)
    lh = ax.errorbar(x, means_p, yerr=sems_p, fmt='o', capsize=5,
                     color='black', ecolor='black', label="Pitch (mean±SEM)")
    ax.set_xticks(x, labels, rotation=45, ha='right')
    ax.set_ylabel("Pitch (nm)")
    ax.set_xlabel("CSV file")
    ax.set_title(f"Pitch: weighted mean ± weighted SEM (σ = pitch_err_nm), R²≥{r2_threshold}")
    lines = [lh]
    labels_legend = ["Pitch (mean±SEM)"]

    # right axis (diameter)
    if diameter:
        means_d = np.empty_like(means_p)
        err_lo_d = np.empty_like(sems_p)
        err_hi_d = np.empty_like(sems_p)

        for i, (mp, sp) in enumerate(zip(means_p, sems_p)):
            if not np.isfinite(mp) or not np.isfinite(sp):
                means_d[i]  = np.nan
                err_lo_d[i] = np.nan
                err_hi_d[i] = np.nan
                continue
        
            d_mid = calculate_diameter(mp)

            p_hi = mp + sp
            p_lo = mp - sp
            try:
                d_hi = calculate_diameter(p_hi)
            except Exception:
                d_hi = np.nan
            try:
                d_lo = calculate_diameter(p_lo)
            except Exception:
                d_lo = np.nan

            means_d[i]  = float(d_mid)
            err_hi_d[i] = float(d_hi - d_mid) if np.isfinite(d_hi) else np.nan
            err_lo_d[i] = float(d_mid - d_lo) if np.isfinite(d_lo) else np.nan

            if np.isfinite(err_hi_d[i]) and err_hi_d[i] < 0:
                err_hi_d[i] = np.nan
            if np.isfinite(err_lo_d[i]) and err_lo_d[i] < 0:
                err_lo_d[i] = np.nan

        ax2 = ax.twinx()
        ax2.tick_params(axis='y', colors='red')
        ax2.yaxis.label.set_color('red')
        yerr_d = np.vstack([err_lo_d, err_hi_d])
        rh = ax2.errorbar(x, means_d, yerr=yerr_d, fmt='s', capsize=5,
                          color='red', label="Diameter (mean±asym err)")
        ax2.set_ylabel("Diameter (nm)")

    plt.tight_layout()
    if save_plot_errorbar:
        plt.savefig(save_plot_errorbar, dpi=300)
    plt.show()

    # histogram
    if len(all_pitch_for_bins) > 0:
        all_concat = np.concatenate(all_pitch_for_bins)
        all_concat = all_concat[np.isfinite(all_concat)]
        if all_concat.size > 0:
            # 共通ビン境界（固定幅）
            min_val, max_val = np.nanmin(all_concat), np.nanmax(all_concat)
            start = np.floor(min_val / BIN_WIDTH_NM) * BIN_WIDTH_NM
            stop  = np.ceil(max_val / BIN_WIDTH_NM) * BIN_WIDTH_NM + BIN_WIDTH_NM*1e-6
            bin_edges = np.arange(start, stop + BIN_WIDTH_NM, BIN_WIDTH_NM)

            n_files = len(per_file_pitch)
            fig, axes = plt.subplots(nrows=n_files, ncols=1,
                                     figsize=(9, 2.6*n_files),
                                     sharex=True, sharey=False)

            if n_files == 1:
                axes = [axes]

            for i, (pvals, lab) in enumerate(zip(per_file_pitch, labels)):
                axh = axes[i]
                if pvals.size == 0:
                    axh.set_visible(False)
                    continue
                color = color_cycle[i % len(color_cycle)]
                axh.hist(pvals, bins=bin_edges, density=False, alpha=0.6,
                         label=lab, color=color, edgecolor='black', linewidth=0.7)
                axh.set_ylabel("Count")
                axh.set_title(f"{lab}  (R²≥{r2_threshold})", fontsize=10)
                # axh.set_xlim(665, 675)

            axes[-1].set_xlabel("Pitch (nm)")

            fig.suptitle(f"Pitch distribution (bin width = {BIN_WIDTH_NM:.2f} nm), R²≥{r2_threshold}",
                         y=0.995, fontsize=12)
            plt.tight_layout()
            if save_plot_hist:
                plt.savefig(save_plot_hist, dpi=300)
            plt.show()
        else:
            print("[WARN] No finite pitch values (after R² filter) for histogram.")
    else:
        print("[WARN] No data passed R² filter to build histogram bins.")

    return summary_df

def load_data(filepath: str, channel: int, t_start: float, t_end: float):
    res = np.load(filepath, allow_pickle=True).item()
    data = res['data']
    sample_rate = res['sample_rate']
    total_samples = res['total_samples']

    t = np.arange(total_samples) / sample_rate

    idx_start = int(t_start * sample_rate)
    idx_end = int(t_end * sample_rate)

    t_sel = t[idx_start:idx_end]
    y_sel = np.array(data[channel][idx_start:idx_end])
    return t_sel, y_sel