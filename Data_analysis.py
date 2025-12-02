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
from analysis2 import *
from beta_vs_radius_plot import CompactFiberSolver

# Three method data analysis

class DataAnalysis:
    
    def __init__(self, filename, N):
        self.filename = filename
        self.N = N
        self.WL = 1389e-9*1e6 # um
        self.total_time = 0.25
        self.percentage_clip = 0.8 # 50% of the data in the center will be used
        self.section_div = 3

    def load_data(self):
        self.t, self.snom = load_data(self.filename, 0, 0, 50)
        self.t, self.mi = load_data(self.filename, 2, 0, 50)
    
    def linear_method(self):
        # Load data
        self.load_data()

        ## For fitting
        def sine_func(t, A, f, phi, offset):
            return A * np.sin(2 * np.pi * f * t + phi) + offset

        def fit_sine_phase_swept(t, y, p0, phase_steps=100):
            best_popt = None
            best_pcov = None
            best_residual = np.inf

            initial_phi_values = np.linspace(0, 2 * np.pi, phase_steps)
            for phi in initial_phi_values:
                p0[2] = phi
                try:
                    popt, pcov = curve_fit(sine_func, t, y, p0=p0)
                    residuals = y - sine_func(t, *popt)
                    ss_res = np.sum(residuals**2)
                    if ss_res < best_residual:
                        best_residual = ss_res
                        best_popt = popt
                        best_pcov = pcov
                except RuntimeError:
                    continue

            return best_popt, best_pcov

        upper_bound = 0.00025
        lower_bound = -0.00025

        total_time = 0.25
        t_center = self.percentage_clip*total_time
        t_partial = (total_time - t_center)/2
        t_start_zc, t_end_zc = self.N + t_partial, self.N + total_time - t_partial

        t_start_zc, t_end_zc = self.N + t_partial, self.N + total_time - t_partial
        mask_zc = (self.t >= t_start_zc) & (self.t <= t_end_zc)
        t_zc_roi, mi_zc_roi = self.t[mask_zc], self.mi[mask_zc]
        mi_zc_roi = (mi_zc_roi - np.min(mi_zc_roi)) / (np.max(mi_zc_roi) - np.min(mi_zc_roi))

        # Find zero points
        zero_indeces = np.where(np.diff(np.sign(mi_zc_roi - 0.5)))[0]
        t_zc_zeros = t_zc_roi[zero_indeces]

        # Find linear region
        dist_2 = []
        diff_2 = []
        for i in range(len(t_zc_zeros)-1):
            dist_2.append(t_zc_zeros[i+1]-t_zc_zeros[i])

        for i in range(len(range(len(dist_2)-1))):
            diff_2.append(dist_2[i+1] - dist_2[i])
        

        indices_less_than_1e_4 = [i for i, d in enumerate(dist_2) if d < 1e-4]

        t_zc_zeros = np.delete(t_zc_zeros, indices_less_than_1e_4)

        # Plot the dist and diffplt.plot(dist_2,'-o', label='dist', markersize=3)
        # plt.plot(diff_2,'-o', label='diff', markersize=3)
        # plt.plot(dist_2,'-o', label='dist', markersize=3)
        # plt.axhline(y=upper_bound, color='r', linestyle='-')
        # plt.axhline(y=lower_bound, color='r', linestyle='-')
        # plt.axhspan(lower_bound, upper_bound, color='red', alpha=0.1, label='')

        # plt.xlabel('Index')
        # plt.ylabel('Time (s)')

        # plt.legend()

        # Find consecutive index in which the two point dist is less than the bound we set
        consecutive_indices = []
        temp = []

        for i in range(len(diff_2)):
            if lower_bound < diff_2[i] < upper_bound:  # strict; use <= if desired
                temp.append(i)
            else:
                if len(temp) >= 9:                     # ≥6 indices
                    consecutive_indices.append(temp)
                temp = []

        # flush the tail run if loop ended inside a streak
        if len(temp) >= 9:
            consecutive_indices.append(temp)
        # print(consecutive_indices)
        ss_res_array = []

        p0 = [0.5, 1/(0.15/20.5), 0, 0.5]  # Initial guess: [Amplitude, Frequency, Phase, Offset]
        try: 
            for i in range(len(consecutive_indices)):
                t_start, t_end = t_zc_zeros[consecutive_indices[i][0]], t_zc_zeros[consecutive_indices[i][-1]]
                mask_fit = (self.t >= t_start) & (self.t <= t_end)
                t_lin_fit_roi, mi_lin_fit_roi = self.t[mask_fit], self.mi[mask_fit]
                mi_lin_fit_roi = (mi_lin_fit_roi - np.min(mi_lin_fit_roi)) / (np.max(mi_lin_fit_roi) - np.min(mi_lin_fit_roi))


                # Implement fitting

                popt, pcov = fit_sine_phase_swept(t_lin_fit_roi, mi_lin_fit_roi, p0, phase_steps=100)
                mi_lin_fit_roi_fit = sine_func(t_lin_fit_roi, *popt)
                residuals = mi_lin_fit_roi - mi_lin_fit_roi_fit
                ss_res = np.sum(residuals**2)
                ss_res_array.append(ss_res)

            # Select the best one
            best_index = np.argmin(ss_res_array)
            t_start, t_end = t_zc_zeros[consecutive_indices[best_index][0]], t_zc_zeros[consecutive_indices[best_index][-1]]
            mask_fit = (self.t >= t_start) & (self.t <= t_end)
            self.t_lin_fit_roi, mi_lin_fit_roi = self.t[mask_fit], self.mi[mask_fit]
            self.mi_lin_fit_roi = (mi_lin_fit_roi - np.min(mi_lin_fit_roi)) / (np.max(mi_lin_fit_roi) - np.min(mi_lin_fit_roi))

            # Best fit
            popt, pcov = fit_sine_phase_swept(self.t_lin_fit_roi, self.mi_lin_fit_roi, p0, phase_steps=100)
            self.mi_lin_fit_roi_fit = sine_func(self.t_lin_fit_roi, *popt)
            residuals = self.mi_lin_fit_roi - self.mi_lin_fit_roi_fit
            ss_res = np.sum(residuals**2)
            
            if np.max(abs(residuals)) < 0.11:
                ## PLOT
                if 0:
                    # --- Plot both in one figure ---
                    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6), gridspec_kw={'height_ratios': [3, 1]})

                    # (1) Data + fit
                    ax1.plot(self.t_lin_fit_roi, self.mi_lin_fit_roi, label="Data", color="tab:orange", marker=".", linestyle="None", markersize=4)
                    ax1.plot(self.t_lin_fit_roi, self.mi_lin_fit_roi_fit, label="Fit", color="k", linewidth=1.2)
                    ax1.set_ylabel("Normalized signal (a.u.)")
                    ax1.legend()
                    ax1.set_title(f'Sample #{self.N}')

                    # (2) Residuals
                    residuals = self.mi_lin_fit_roi - self.mi_lin_fit_roi_fit
                    ss_res = np.sum(residuals**2)

                    ax2.plot(self.t_lin_fit_roi, residuals, label="Residuals", color="red", marker=".", linestyle="None", markersize=4)
                    ax2.axhline(0, color="k", linestyle="--", linewidth=0.8)
                    ax2.set_xlabel("Time (s)")
                    ax2.set_ylabel("Residual")
                    ax2.legend()
                    plt.tight_layout()
                    plt.show()

                # Convert time to position
                f_opt = popt[1] # /s
                dx_dt = self.WL/2 * f_opt # um/s


                self.x_linear = (self.t_lin_fit_roi - self.t_lin_fit_roi[0]) * dx_dt
                snom_fit_roi = self.snom[mask_fit]
                self.snom_linear = (snom_fit_roi - np.min(snom_fit_roi)) / (np.max(snom_fit_roi) - np.min(snom_fit_roi))

                ## PLOT
                if 0:
                    plt.plot(self.x_linear, self.snom_linear, label="snom", marker = '.')
                    plt.xlabel("Position (um)")
                    plt.ylabel("Normalized signal (a.u.)")
                    plt.legend()
                    plt.show()
                
                return self.x_linear, self.snom_linear
        except:
            pass

    def nonlinear_method(self):
        # ---- quick FFT freq guess (robust to uneven sampling via resampling) ----
        def f_guess(x, y):
            n = min(4096, max(256, 2**int(np.ceil(np.log2(len(x))))))
            xu = np.linspace(x.min(), x.max(), n)
            yu = np.interp(xu, x, y - np.median(y))
            dt = np.median(np.diff(xu))
            Y = np.fft.rfft(yu)
            f = np.fft.rfftfreq(len(yu), d=dt)
            if len(f) <= 2:
                return 1.0 / max(x.max()-x.min(), 1.0)
            k = 1 + np.argmax(np.abs(Y[1:]))  # ignore DC
            return max(float(f[k]), 1e-9)

        # ---- model factory: captures m and x0 cleanly ----
        def make_model(m, x0):
            def model(x, A, f, phi0, *c_and_offset):
                c = c_and_offset[:-1]; off = c_and_offset[-1]
                xt = x - x0
                if m > 0:
                    # P = Σ_{k=1..m} c_k xt^k ; build via Vandermonde
                    P = np.vander(xt, N=m+1, increasing=True)[:, 1:] @ np.asarray(c)
                else:
                    P = 0.0
                return off + A * np.sin(2*np.pi*f*xt + phi0 + P)
            return model

        # ---- instantaneous frequency from fitted params ----
        def finst_from_params(xq, m, x0, f_hz, c):
            xt = xq - x0
            dphi_dx = 2*np.pi*f_hz
            if m > 0 and len(c) == m:
                V = np.vander(xt, N=m, increasing=True)          # xt^0..xt^{m-1}
                kc = np.arange(1, m+1) * np.asarray(c)           # k*c_k
                dphi_dx = dphi_dx + V @ kc
            return dphi_dx / (2*np.pi)
        
        # load data 
        self.load_data()

        # clip the data 
        total_time = self.total_time
        t_center = self.percentage_clip*total_time
        t_partial = (total_time - t_center)/2
        t_start_pol, t_end_pol = self.N + t_partial, self.N + total_time - t_partial

        mask_pol = (self.t >= t_start_pol) & (self.t <= t_end_pol)
        t_pol_roi, mi_pol_roi, snom_pol_roi= self.t[mask_pol], self.mi[mask_pol],self.snom[mask_pol]
        mi_pol_roi = (mi_pol_roi - np.min(mi_pol_roi)) / (np.max(mi_pol_roi) - np.min(mi_pol_roi))
        snom_pol_roi_norm = (snom_pol_roi - np.min(snom_pol_roi)) / (np.max(snom_pol_roi) - np.min(snom_pol_roi))

        # ---- sweep small phase orders and pick the best ----
        x = np.asarray(t_pol_roi, float).ravel()
        y = np.asarray(mi_pol_roi, float).ravel()
        x0 = x.mean()

        A0 = 0.5*(np.nanpercentile(y, 95) - np.nanpercentile(y, 5))
        f0 = f_guess(x, y)
        m_values = range(15)   # keep small to avoid overfit/instability

        results = []              # will hold dicts per m
        Res_array = []            # SSE per m for selection

        for m in m_values:
            model = make_model(m, x0)
            p0 = [A0, f0, 0.0] + [0.0]*m + [np.median(y)]             # [A,f,phi0,c1..cm,off]
            lb = [0.0, 1e-9, -np.inf] + [-np.inf]*m + [-np.inf]
            ub = [np.inf,  np.inf,  np.inf] + [ np.inf]*m + [ np.inf]
            try:
                popt, pcov = curve_fit(model, x, y, p0=p0, bounds=(lb, ub), maxfev=20000)
                yhat = model(x, *popt)
                ss_res = float(np.sum((y - yhat)**2))
                Res_array.append(ss_res)
                results.append({"m": m, "popt": popt, "pcov": pcov, "ss_res": ss_res})
            except Exception:
                Res_array.append(np.inf)
                results.append({"m": m, "popt": None, "pcov": None, "ss_res": np.inf})
        

        # ---- choose best order ----
        best_idx = int(np.argmin(Res_array))
        best = results[best_idx]
        m_best = best["m"]
        popt = best["popt"]
        assert popt is not None, "All fits failed; consider smoothing data or narrowing m_values."

        # ---- unpack best params ----
        A, f_hz, phi0 = popt[0], popt[1], popt[2]
        c = popt[3:3+m_best] if m_best > 0 else []
        offset = popt[-1]

        # ---- smooth line + instantaneous frequency ----
        xd = np.linspace(x.min(), x.max(), 2000)
        model_best = make_model(m_best, x0)
        yd = model_best(xd, *popt)
        res = y - model_best(x, *popt)
        f_inst = finst_from_params(xd, m_best, x0, f_hz, c)

        # print(f"Best order m = {m_best}  |  SSE = {best['ss_res']:.6g}")
        # print(f"A={A:.6g}, f={f_hz:.6g} Hz, phi0={phi0:.6g}, offset={offset:.6g}")
        # for i, ck in enumerate(c, 1):
            # print(f"c{i}={ck:.6g}")

        # ---- (optional) plots: data+fit, residuals, instantaneous frequency ----
        if 0:
            fig = plt.figure(figsize=(12, 10))
            gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.2, 1.8], hspace=0.22)
            ax1 = fig.add_subplot(gs[0,0]); ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
            ax1.plot(x, y, '.', ms=3, label='Data')
            ax1.plot(xd, yd, '-', lw=0.5, label=f'Fit (m={m_best})',color='k')
            ax1.set_ylabel('MI signal (a.u.)')
            ax1.set_title(f"Sample # {self.N}")
            ax1.legend()

            ax2.plot(x, res, '.', ms=3, label='Residuals', color='C3')
            ax2.axhline(0, ls='--', lw=0.8, color='k')
            ax2.set_ylabel('Residual')
            # ax2.legend()
            # ax2.set_title(f"SSE = {best['ss_res']:.3g}")

            plt.tight_layout(); plt.show()

        # inputs from your fit
        t_ = t_pol_roi.astype(float)
        t0 = t_.mean()
        A, f, phi0 = popt[0], popt[1], popt[2]
        m = len(popt) - 4                     # number of phase poly terms
        c = np.asarray(popt[3:3+m]) if m>0 else np.array([])
        offset = popt[-1]

        # constants
        lam = self.WL * 1e-3   # 1389 nm
        n = 1.0           # ~1.00027 in air if you want to be precise

        tt = t_ - t0
        # phase(t):
        phi = 2*np.pi*f*tt + phi0 + (np.vander(tt, N=m+1, increasing=True)[:,1:] @ c if m>0 else 0.0)

        # choose reference so x(t0)=0 => subtract phi0
        x = (lam/(4*np.pi*n)) * (phi - phi0)          # position (meters)
        x_lin = x - x[0]
        # instantaneous frequency and velocity
        # dphi/dt = 2π f + Σ k c_k (t-t0)^{k-1}
        if m>0:
            V = np.vander(tt, N=m, increasing=True)   # columns: tt^0 .. tt^{m-1}
            kc = (np.arange(1, m+1) * c)              # k * c_k
            dphi_dt = 2*np.pi*f + V @ kc
        else:
            dphi_dt = 2*np.pi*f * np.ones_like(tt)

        f_inst = dphi_dt / (2*np.pi)                  # Hz
        v = (lam/(2*n)) * f_inst  

        self.x_nonlinear = x_lin*1e3
        self.snom_nonlinear = snom_pol_roi_norm  

        return self.x_nonlinear, self.snom_nonlinear

    def nonlinear_method_by_section(self):
        # ---- quick FFT freq guess (robust to uneven sampling via resampling) ----
        def f_guess(x, y):
            n = min(4096, max(256, 2**int(np.ceil(np.log2(len(x))))))
            xu = np.linspace(x.min(), x.max(), n)
            yu = np.interp(xu, x, y - np.median(y))
            dt = np.median(np.diff(xu))
            Y = np.fft.rfft(yu)
            f = np.fft.rfftfreq(len(yu), d=dt)
            if len(f) <= 2:
                return 1.0 / max(x.max()-x.min(), 1.0)
            k = 1 + np.argmax(np.abs(Y[1:]))  # ignore DC
            return max(float(f[k]), 1e-9)

        # ---- model factory: captures m and x0 cleanly ----
        def make_model(m, x0):
            def model(x, A, f, phi0, *c_and_offset):
                c = c_and_offset[:-1]; off = c_and_offset[-1]
                xt = x - x0
                if m > 0:
                    # P = Σ_{k=1..m} c_k xt^k ; build via Vandermonde
                    P = np.vander(xt, N=m+1, increasing=True)[:, 1:] @ np.asarray(c)
                else:
                    P = 0.0
                return off + A * np.sin(2*np.pi*f*xt + phi0 + P)
            return model

        # ---- instantaneous frequency from fitted params ----
        def finst_from_params(xq, m, x0, f_hz, c):
            xt = xq - x0
            dphi_dx = 2*np.pi*f_hz
            if m > 0 and len(c) == m:
                V = np.vander(xt, N=m, increasing=True)          # xt^0..xt^{m-1}
                kc = np.arange(1, m+1) * np.asarray(c)           # k*c_k
                dphi_dx = dphi_dx + V @ kc
            return dphi_dx / (2*np.pi)
        
        # load data 
        self.load_data()

        # clip the data 
        total_time = self.total_time
        t_center = self.percentage_clip*total_time
        t_partial = (total_time - t_center)/2
        t_section = t_center/self.section_div
        t_start_pol_1, t_end_pol_1 = self.N + t_partial, self.N + t_section + t_partial
        t_start_pol_2, t_end_pol_2 = self.N + t_section + t_partial, self.N + 2*t_section + t_partial
        t_start_pol_3, t_end_pol_3 = self.N + 2*t_section + t_partial, self.N + 3*t_section + t_partial

        mask_pol_1 = (self.t >= t_start_pol_1) & (self.t <= t_end_pol_1)
        mask_pol_2 = (self.t >= t_start_pol_2) & (self.t <= t_end_pol_2)
        mask_pol_3 = (self.t >= t_start_pol_3) & (self.t <= t_end_pol_3)

        t_pol_roi_1, mi_pol_roi_1, snom_pol_roi_1= self.t[mask_pol_1], self.mi[mask_pol_1],self.snom[mask_pol_1]
        mi_pol_roi_1 = (mi_pol_roi_1 - np.min(mi_pol_roi_1)) / (np.max(mi_pol_roi_1) - np.min(mi_pol_roi_1))
        snom_pol_roi_1_norm = (snom_pol_roi_1 - np.min(snom_pol_roi_1)) / (np.max(snom_pol_roi_1) - np.min(snom_pol_roi_1))

        t_pol_roi_2, mi_pol_roi_2, snom_pol_roi_2= self.t[mask_pol_2], self.mi[mask_pol_2],self.snom[mask_pol_2]
        mi_pol_roi_2 = (mi_pol_roi_2 - np.min(mi_pol_roi_2)) / (np.max(mi_pol_roi_2) - np.min(mi_pol_roi_2))
        snom_pol_roi_2_norm = (snom_pol_roi_2 - np.min(snom_pol_roi_2)) / (np.max(snom_pol_roi_2) - np.min(snom_pol_roi_2))

        t_pol_roi_3, mi_pol_roi_3, snom_pol_roi_3= self.t[mask_pol_3], self.mi[mask_pol_3],self.snom[mask_pol_3]
        mi_pol_roi_3 = (mi_pol_roi_3 - np.min(mi_pol_roi_3)) / (np.max(mi_pol_roi_3) - np.min(mi_pol_roi_3))
        snom_pol_roi_3_norm = (snom_pol_roi_3 - np.min(snom_pol_roi_3)) / (np.max(snom_pol_roi_3) - np.min(snom_pol_roi_3))

        # ---- sweep small phase orders and pick the best ----
        x_1 = np.asarray(t_pol_roi_1, float).ravel()
        y_1 = np.asarray(mi_pol_roi_1, float).ravel()
        x0_1 = x_1.mean()

        A0_1 = 0.5*(np.nanpercentile(y_1, 95) - np.nanpercentile(y_1, 5))
        f0_1 = f_guess(x_1, y_1)

        x_2 = np.asarray(t_pol_roi_2, float).ravel()
        y_2 = np.asarray(mi_pol_roi_2, float).ravel()
        x0_2 = x_2.mean()

        A0_2 = 0.5*(np.nanpercentile(y_2, 95) - np.nanpercentile(y_2, 5))
        f0_2 = f_guess(x_2, y_2)

        x_3 = np.asarray(t_pol_roi_3, float).ravel()
        y_3 = np.asarray(mi_pol_roi_3, float).ravel()
        x0_3 = x_3.mean()

        A0_3 = 0.5*(np.nanpercentile(y_3, 95) - np.nanpercentile(y_3, 5))
        f0_3 = f_guess(x_3, y_3)

        m_values = range(8)   # keep small to avoid overfit/instability

        results_1 = []              # will hold dicts per m
        results_2 = []              # will hold dicts per m
        results_3 = []              # will hold dicts per m

        Res_array_1 = []            # SSE per m for selection
        Res_array_2 = []            # SSE per m for selection
        Res_array_3 = []            # SSE per m for selection

        for m in m_values:
            model_1 = make_model(m, x0_1)
            model_2 = make_model(m, x0_2)
            model_3 = make_model(m, x0_3)
            p0_1 = [A0_1, f0_1, 0.0] + [0.0]*m + [np.median(y_1)]             # [A,f,phi0,c1..cm,off]
            p0_2 = [A0_2, f0_2, 0.0] + [0.0]*m + [np.median(y_2)]             # [A,f,phi0,c1..cm,off]
            p0_3 = [A0_3, f0_3, 0.0] + [0.0]*m + [np.median(y_3)]             # [A,f,phi0,c1..cm,off]
            lb_1 = [0.0, 1e-9, -np.inf] + [-np.inf]*m + [-np.inf]
            lb_2 = [0.0, 1e-9, -np.inf] + [-np.inf]*m + [-np.inf]
            lb_3 = [0.0, 1e-9, -np.inf] + [-np.inf]*m + [-np.inf]
            ub_1 = [np.inf,  np.inf,  np.inf] + [ np.inf]*m + [ np.inf]
            ub_2 = [np.inf,  np.inf,  np.inf] + [ np.inf]*m + [ np.inf]
            ub_3 = [np.inf,  np.inf,  np.inf] + [ np.inf]*m + [ np.inf]
            try:
                popt_1, pcov_1 = curve_fit(model_1, x_1, y_1, p0=p0_1, bounds=(lb_1, ub_1), maxfev=20000)
                popt_2, pcov_2 = curve_fit(model_2, x_2, y_2, p0=p0_2, bounds=(lb_2, ub_2), maxfev=20000)
                popt_3, pcov_3 = curve_fit(model_3, x_3, y_3, p0=p0_3, bounds=(lb_3, ub_3), maxfev=20000)

                yhat_1 = model_1(x_1, *popt_1)
                yhat_2 = model_2(x_2, *popt_2)
                yhat_3 = model_3(x_3, *popt_3)

                ss_res_1 = float(np.sum((y_1 - yhat_1)**2))
                ss_res_2 = float(np.sum((y_2 - yhat_2)**2))
                ss_res_3 = float(np.sum((y_3 - yhat_3)**2))

                Res_array_1.append(ss_res_1)
                Res_array_2.append(ss_res_2)
                Res_array_3.append(ss_res_3)

                results_1.append({"m": m, "popt": popt_1, "pcov": pcov_1, "ss_res": ss_res_1})
                results_2.append({"m": m, "popt": popt_2, "pcov": pcov_2, "ss_res": ss_res_2})
                results_3.append({"m": m, "popt": popt_3, "pcov": pcov_3, "ss_res": ss_res_3})

            except Exception:
                Res_array_1.append(np.inf)
                Res_array_2.append(np.inf)
                Res_array_3.append(np.inf)
                results_1.append({"m": m, "popt": None, "pcov": None, "ss_res": np.inf})
                results_2.append({"m": m, "popt": None, "pcov": None, "ss_res": np.inf})

        # ---- choose best order ----
        best_idx_1 = int(np.argmin(Res_array_1))
        best_idx_2 = int(np.argmin(Res_array_2))
        best_idx_3 = int(np.argmin(Res_array_3))
        best_1 = results_1[best_idx_1]
        best_2 = results_2[best_idx_2]
        best_3 = results_3[best_idx_3]
        m_best_1 = best_1["m"]
        m_best_2 = best_2["m"]
        m_best_3 = best_3["m"]
        popt_1 = best_1["popt"]
        popt_2 = best_2["popt"]
        popt_3 = best_3["popt"]
        assert popt_1 is not None, "All fits failed; consider smoothing data or narrowing m_values."
        assert popt_2 is not None, "All fits failed; consider smoothing data or narrowing m_values."
        assert popt_3 is not None, "All fits failed; consider smoothing data or narrowing m_values."

        # ---- unpack best params ----
        A_1, f_hz_1, phi0_1 = popt_1[0], popt_1[1], popt_1[2]
        A_2, f_hz_2, phi0_2 = popt_2[0], popt_2[1], popt_2[2]
        A_3, f_hz_3, phi0_3 = popt_3[0], popt_3[1], popt_3[2]
        c_1 = popt_1[3:3+m_best_1] if m_best_1 > 0 else []
        c_2 = popt_2[3:3+m_best_2] if m_best_2 > 0 else []
        c_3 = popt_3[3:3+m_best_3] if m_best_3 > 0 else []
        offset_1 = popt_1[-1]
        offset_2 = popt_2[-1]
        offset_3 = popt_3[-1]

        # ---- smooth line + instantaneous frequency ----
        xd_1 = np.linspace(x_1.min(), x_1.max(), 2000)
        xd_2 = np.linspace(x_2.min(), x_2.max(), 2000)
        xd_3 = np.linspace(x_3.min(), x_3.max(), 2000)

        model_best_1 = make_model(m_best_1, x0_1)
        model_best_2 = make_model(m_best_2, x0_2)
        model_best_3 = make_model(m_best_3, x0_3)

        yd_1 = model_best_1(xd_1, *popt_1)
        yd_2 = model_best_2(xd_2, *popt_2)
        yd_3 = model_best_3(xd_3, *popt_3)

        res_1 = y_1 - model_best_1(x_1, *popt_1)
        res_2 = y_2 - model_best_2(x_2, *popt_2)
        res_3 = y_3 - model_best_3(x_3, *popt_3)

        f_inst_1 = finst_from_params(xd_1, m_best_1, x0_1, f_hz_1, c_1)
        f_inst_2 = finst_from_params(xd_2, m_best_2, x0_2, f_hz_2, c_2)
        f_inst_3 = finst_from_params(xd_3, m_best_3, x0_3, f_hz_3, c_3)

        # ---- (optional) plots: data+fit, residuals, instantaneous frequency ----
        if 0:
            fig = plt.figure(figsize=(12, 10))
            gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.2, 1.8], hspace=0.22)
            ax1 = fig.add_subplot(gs[0,0]); ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
            ax1.plot(x_1, y_1, '.', ms=3, label='Data')
            ax1.plot(xd_1, yd_1, '-', lw=0.5, label=f'Fit (m={m_best_1})',color='k')
            ax1.set_ylabel('MI signal (a.u.)')
            ax1.set_title(f"Sample # {self.N}")
            ax1.legend()

            ax2.plot(x_1, res_1, '.', ms=3, label='Residuals', color='C3')
            ax2.axhline(0, ls='--', lw=0.8, color='k')
            ax2.set_ylabel('Residual')
            # ax2.legend()
            # ax2.set_title(f"SSE = {best['ss_res']:.3g}")

            plt.tight_layout(); plt.show()

            fig = plt.figure(figsize=(12, 10))
            gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.2, 1.8], hspace=0.22)
            ax1 = fig.add_subplot(gs[0,0]); ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
            ax1.plot(x_2, y_2, '.', ms=3, label='Data')
            ax1.plot(xd_2, yd_2, '-', lw=0.5, label=f'Fit (m={m_best_2})',color='k')
            ax1.set_ylabel('MI signal (a.u.)')
            ax1.set_title(f"Sample # {self.N}")
            ax1.legend()
            
            ax2.plot(x_2, res_2, '.', ms=3, label='Residuals', color='C3')
            ax2.axhline(0, ls='--', lw=0.8, color='k')
            ax2.set_ylabel('Residual')
            plt.tight_layout(); plt.show()

            fig = plt.figure(figsize=(12, 10))
            gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.2, 1.8], hspace=0.22)
            ax1 = fig.add_subplot(gs[0,0]); ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
            ax1.plot(x_3, y_3, '.', ms=3, label='Data')
            ax1.plot(xd_3, yd_3, '-', lw=0.5, label=f'Fit (m={m_best_3})',color='k')
            ax1.set_ylabel('MI signal (a.u.)')
            ax1.set_title(f"Sample # {self.N}")
            ax1.legend()

            ax2.plot(x_3, res_3, '.', ms=3, label='Residuals', color='C3')
            ax2.axhline(0, ls='--', lw=0.8, color='k')
            ax2.set_ylabel('Residual')
            plt.tight_layout(); plt.show()

        # inputs from your fit
        t_1 = t_pol_roi_1.astype(float)
        t0_1 = t_1.mean()
        A_1, f_1, phi0_1 = popt_1[0], popt_1[1], popt_1[2]
        m_1 = len(popt_1) - 4                     # number of phase poly terms
        c_1 = np.asarray(popt_1[3:3+m_1]) if m_1>0 else np.array([])
        offset_1 = popt_1[-1]

        t_2 = t_pol_roi_2.astype(float)
        t0_2 = t_2.mean()
        A_2, f_2, phi0_2 = popt_2[0], popt_2[1], popt_2[2]
        m_2 = len(popt_2) - 4                     # number of phase poly terms
        c_2 = np.asarray(popt_2[3:3+m_2]) if m_2>0 else np.array([])
        offset_2 = popt_2[-1]

        t_3 = t_pol_roi_3.astype(float)
        t0_3 = t_3.mean()
        A_3, f_3, phi0_3 = popt_3[0], popt_3[1], popt_3[2]
        m_3 = len(popt_3) - 4                     # number of phase poly terms
        c_3 = np.asarray(popt_3[3:3+m_3]) if m_3>0 else np.array([])
        offset_3 = popt_3[-1]

        # constants
        lam = self.WL * 1e-3   # 1389 nm
        n = 1.0           # ~1.00027 in air if you want to be precise

        tt_1 = t_1 - t0_1
        tt_2 = t_2 - t0_2
        tt_3 = t_3 - t0_3
        # phase(t):
        phi_1 = 2*np.pi*f_1*tt_1 + phi0_1 + (np.vander(tt_1, N=m_1+1, increasing=True)[:,1:] @ c_1 if m_1>0 else 0.0)
        phi_2 = 2*np.pi*f_2*tt_2 + phi0_2 + (np.vander(tt_2, N=m_2+1, increasing=True)[:,1:] @ c_2 if m_2>0 else 0.0)
        phi_3 = 2*np.pi*f_3*tt_3 + phi0_3 + (np.vander(tt_3, N=m_3+1, increasing=True)[:,1:] @ c_3 if m_3>0 else 0.0)

        # choose reference so x(t0)=0 => subtract phi0
        x_1 = (lam/(4*np.pi*n)) * (phi_1 - phi0_1)          # position (meters)
        x_2 = (lam/(4*np.pi*n)) * (phi_2 - phi0_2)          # position (meters)
        x_3 = (lam/(4*np.pi*n)) * (phi_3 - phi0_3)          # position (meters)
        x_lin_1 = x_1 - x_1[0]
        x_lin_2 = x_2 - x_2[0]
        x_lin_3 = x_3 - x_3[0]
        # instantaneous frequency and velocity
        # dphi/dt = 2π f + Σ k c_k (t-t0)^{k-1}
        if m>0:
            V_1 = np.vander(tt_1, N=m_1, increasing=True)   # columns: tt^0 .. tt^{m-1}
            V_2 = np.vander(tt_2, N=m_2, increasing=True)   # columns: tt^0 .. tt^{m-1}
            V_3 = np.vander(tt_3, N=m_3, increasing=True)   # columns: tt^0 .. tt^{m-1}
            kc_1 = (np.arange(1, m_1+1) * c_1)              # k * c_k
            kc_2 = (np.arange(1, m_2+1) * c_2)              # k * c_k
            kc_3 = (np.arange(1, m_3+1) * c_3)              # k * c_k
            dphi_dt_1 = 2*np.pi*f_1 + V_1 @ kc_1
            dphi_dt_2 = 2*np.pi*f_2 + V_2 @ kc_2
            dphi_dt_3 = 2*np.pi*f_3 + V_3 @ kc_3
        else:
            dphi_dt_1 = 2*np.pi*f_1 * np.ones_like(tt_1)
            dphi_dt_2 = 2*np.pi*f_2 * np.ones_like(tt_2)
            dphi_dt_3 = 2*np.pi*f_3 * np.ones_like(tt_3)

        f_inst_1 = dphi_dt_1 / (2*np.pi)                  # Hz
        f_inst_2 = dphi_dt_2 / (2*np.pi)                  # Hz
        f_inst_3 = dphi_dt_3 / (2*np.pi)                  # Hz
        v_1 = (lam/(2*n)) * f_inst_1  
        v_2 = (lam/(2*n)) * f_inst_2  
        v_3 = (lam/(2*n)) * f_inst_3  

        xxx_1 = x_lin_1*1e3 
        xxx_2 = x_lin_2*1e3 
        xxx_3 = x_lin_3*1e3 
        
        yyy_1 = snom_pol_roi_1_norm  
        yyy_2 = snom_pol_roi_2_norm  
        yyy_3 = snom_pol_roi_3_norm  

        return xxx_1, yyy_1, xxx_2, yyy_2, xxx_3, yyy_3

    def nonlinear_method_sim(self, t, mi):
        # ---- quick FFT freq guess (robust to uneven sampling via resampling) ----
        def f_guess(x, y):
            n = min(4096, max(256, 2**int(np.ceil(np.log2(len(x))))))
            xu = np.linspace(x.min(), x.max(), n)
            yu = np.interp(xu, x, y - np.median(y))
            dt = np.median(np.diff(xu))
            Y = np.fft.rfft(yu)
            f = np.fft.rfftfreq(len(yu), d=dt)
            if len(f) <= 2:
                return 1.0 / max(x.max()-x.min(), 1.0)
            k = 1 + np.argmax(np.abs(Y[1:]))  # ignore DC
            return max(float(f[k]), 1e-9)

        # ---- model factory: captures m and x0 cleanly ----
        def make_model(m, x0):
            def model(x, A, f, phi0, *c_and_offset):
                c = c_and_offset[:-1]; off = c_and_offset[-1]
                xt = x - x0
                if m > 0:
                    # P = Σ_{k=1..m} c_k xt^k ; build via Vandermonde
                    P = np.vander(xt, N=m+1, increasing=True)[:, 1:] @ np.asarray(c)
                else:
                    P = 0.0
                return off + A * np.sin(2*np.pi*f*xt + phi0 + P)
            return model

        # ---- instantaneous frequency from fitted params ----
        def finst_from_params(xq, m, x0, f_hz, c):
            xt = xq - x0
            dphi_dx = 2*np.pi*f_hz
            if m > 0 and len(c) == m:
                V = np.vander(xt, N=m, increasing=True)          # xt^0..xt^{m-1}
                kc = np.arange(1, m+1) * np.asarray(c)           # k*c_k
                dphi_dx = dphi_dx + V @ kc
            return dphi_dx / (2*np.pi)
        
        t_pol_roi, mi_pol_roi = t, mi

        # ---- sweep small phase orders and pick the best ----
        x = np.asarray(t_pol_roi, float).ravel()
        y = np.asarray(mi_pol_roi, float).ravel()
        x0 = x.mean()

        A0 = 0.5*(np.nanpercentile(y, 95) - np.nanpercentile(y, 5))
        f0 = f_guess(x, y)
        m_values = [15]   # keep small to avoid overfit/instability

        results = []              # will hold dicts per m
        Res_array = []            # SSE per m for selection

        for m in m_values:
            model = make_model(m, x0)
            p0 = [A0, f0, 0.0] + [0.0]*m + [np.median(y)]             # [A,f,phi0,c1..cm,off]
            lb = [0.0, 1e-9, -np.inf] + [-np.inf]*m + [-np.inf]
            ub = [np.inf,  np.inf,  np.inf] + [ np.inf]*m + [ np.inf]
            try:
                popt, pcov = curve_fit(model, x, y, p0=p0, bounds=(lb, ub), maxfev=20000)
                yhat = model(x, *popt)
                ss_res = float(np.sum((y - yhat)**2))
                Res_array.append(ss_res)
                results.append({"m": m, "popt": popt, "pcov": pcov, "ss_res": ss_res})
            except Exception:
                Res_array.append(np.inf)
                results.append({"m": m, "popt": None, "pcov": None, "ss_res": np.inf})
        
        # ---- choose best order ----
        best_idx = int(np.argmin(Res_array))
        best = results[best_idx]
        m_best = best["m"]
        popt = best["popt"]
        assert popt is not None, "All fits failed; consider smoothing data or narrowing m_values."

        # ---- unpack best params ----
        A, f_hz, phi0 = popt[0], popt[1], popt[2]
        c = popt[3:3+m_best] if m_best > 0 else []
        offset = popt[-1]

        # ---- smooth line + instantaneous frequency ----
        xd = np.linspace(x.min(), x.max(), 2000)
        model_best = make_model(m_best, x0)
        yd = model_best(xd, *popt)
        res = y - model_best(x, *popt)
        f_inst = finst_from_params(xd, m_best, x0, f_hz, c)

        # print(f"Best order m = {m_best}  |  SSE = {best['ss_res']:.6g}")
        # print(f"A={A:.6g}, f={f_hz:.6g} Hz, phi0={phi0:.6g}, offset={offset:.6g}")
        # for i, ck in enumerate(c, 1):
            # print(f"c{i}={ck:.6g}")

        # ---- (optional) plots: data+fit, residuals, instantaneous frequency ----
        if 0:
            fig = plt.figure(figsize=(12, 10))
            gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.2, 1.8], hspace=0.22)
            ax1 = fig.add_subplot(gs[0,0]); ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
            ax1.plot(x, y, '.', ms=3, label='Data')
            ax1.plot(xd, yd, '-', lw=0.5, label=f'Fit (m={m_best})',color='k')
            ax1.set_ylabel('MI signal (a.u.)')
            ax1.set_title(f"Sample # {self.N}")
            ax1.legend()

            ax2.plot(x, res, '.', ms=3, label='Residuals', color='C3')
            ax2.axhline(0, ls='--', lw=0.8, color='k')
            ax2.set_ylabel('Residual')
            # ax2.legend()
            # ax2.set_title(f"SSE = {best['ss_res']:.3g}")

            plt.tight_layout(); plt.show()

        # inputs from your fit
        t_ = t_pol_roi.astype(float)
        t0 = t_.mean()
        A, f, phi0 = popt[0], popt[1], popt[2]
        m = len(popt) - 4                     # number of phase poly terms
        c = np.asarray(popt[3:3+m]) if m>0 else np.array([])
        offset = popt[-1]

        # constants
        lam = self.WL * 1e-3   # 1389 nm
        n = 1.0           # ~1.00027 in air if you want to be precise

        tt = t_ - t0
        # phase(t):
        phi = 2*np.pi*f*tt + phi0 + (np.vander(tt, N=m+1, increasing=True)[:,1:] @ c if m>0 else 0.0)

        # choose reference so x(t0)=0 => subtract phi0
        x = (lam/(4*np.pi*n)) * (phi - phi0)          # position (meters)
        x_lin = x - x[0]
        # instantaneous frequency and velocity
        # dphi/dt = 2π f + Σ k c_k (t-t0)^{k-1}
        if m>0:
            V = np.vander(tt, N=m, increasing=True)   # columns: tt^0 .. tt^{m-1}
            kc = (np.arange(1, m+1) * c)              # k * c_k
            dphi_dt = 2*np.pi*f + V @ kc
        else:
            dphi_dt = 2*np.pi*f * np.ones_like(tt)

        f_inst = dphi_dt / (2*np.pi)                  # Hz
        v = (lam/(2*n)) * f_inst  

        self.x_nonlinear = x_lin*1e3

        return xd, yd

    def zero_crossing_interpol(self):
        # load data
        self.load_data()

        # clip the data and only taking 80% of the entire data
        total_time = self.total_time
        t_center = self.percentage_clip*total_time
        t_partial = (total_time - t_center)/2
        t_start_zc, t_end_zc = self.N + t_partial, self.N + total_time - t_partial
        mask_zc = (self.t >= t_start_zc) & (self.t <= t_end_zc)
        t_zc_roi, mi_zc_roi = self.t[mask_zc], self.mi[mask_zc]
        mi_zc_roi = (mi_zc_roi - np.min(mi_zc_roi)) / (np.max(mi_zc_roi) - np.min(mi_zc_roi))

        # Find zero points
        zero_indeces = np.where(np.diff(np.sign(mi_zc_roi - 0.5)))[0]
        t_zc_zeros = t_zc_roi[zero_indeces]

        # Filter double crossing points
        dist_2 = []
        for i in range(len(t_zc_zeros)-1):
            dist_2.append(t_zc_zeros[i+1]-t_zc_zeros[i])
        

        indices_less_than_1e_4 = [i for i, d in enumerate(dist_2) if d < 1e-4]

        t_zc_zeros = np.delete(t_zc_zeros, indices_less_than_1e_4)

        # Time to position conversion by linear interpolating the zero crossing points
        WL = self.WL
        wl_ref = np.arange(len(t_zc_zeros))
        x_zc_roi = WL/4 * np.interp(t_zc_roi, t_zc_zeros, wl_ref)

        # Identify flat edges on both sides
        edge_tolerance = 1e-6  # Define a tolerance for flatness
        deriv = np.gradient(x_zc_roi)
        
        # Find flat regions at the beginning
        start_flat = np.argmax(deriv > edge_tolerance)
        
        # Find flat regions at the end
        end_flat = len(deriv) - np.argmax(deriv[::-1] > edge_tolerance)

        # Remove flat edges
        if start_flat < end_flat:
            x_zc_roi = x_zc_roi[start_flat:end_flat]
            

        # SNOM signal normalization
        snom_zc_roi = self.snom[mask_zc]
        snom_zc_roi = (snom_zc_roi - np.min(snom_zc_roi)) / (np.max(snom_zc_roi) - np.min(snom_zc_roi))
        snom_zc_roi = snom_zc_roi[start_flat:end_flat]

        self.x_lin_interpol = x_zc_roi
        self.snom_lin_interpol = snom_zc_roi

        return

    def zero_crossing_interpol_by_section(self):
        # load data
        self.load_data()

        # clip the data 
        total_time = self.total_time
        center_time = self.percentage_clip*total_time
        t_partial = (total_time - center_time)/2
        t_section = center_time/self.section_div

        t_start_zc_1, t_end_zc_1 = self.N + t_partial, self.N + t_section + t_partial
        t_start_zc_2, t_end_zc_2 = self.N + t_section + t_partial, self.N + 2*t_section + t_partial
        t_start_zc_3, t_end_zc_3 = self.N + 2*t_section + t_partial, self.N + 3*t_section + t_partial

        mask_zc_1 = (self.t >= t_start_zc_1) & (self.t <= t_end_zc_1)
        mask_zc_2 = (self.t >= t_start_zc_2) & (self.t <= t_end_zc_2)
        mask_zc_3 = (self.t >= t_start_zc_3) & (self.t <= t_end_zc_3)

        t_zc_roi_1, mi_zc_roi_1 = self.t[mask_zc_1], self.mi[mask_zc_1]
        t_zc_roi_2, mi_zc_roi_2 = self.t[mask_zc_2], self.mi[mask_zc_2]
        t_zc_roi_3, mi_zc_roi_3 = self.t[mask_zc_3], self.mi[mask_zc_3]

        mi_zc_roi_1 = (mi_zc_roi_1 - np.min(mi_zc_roi_1)) / (np.max(mi_zc_roi_1) - np.min(mi_zc_roi_1))
        mi_zc_roi_2 = (mi_zc_roi_2 - np.min(mi_zc_roi_2)) / (np.max(mi_zc_roi_2) - np.min(mi_zc_roi_2))
        mi_zc_roi_3 = (mi_zc_roi_3 - np.min(mi_zc_roi_3)) / (np.max(mi_zc_roi_3) - np.min(mi_zc_roi_3))

        # Find zero points for each section
        zero_indeces_1 = np.where(np.diff(np.sign(mi_zc_roi_1 - 0.5)))[0]
        zero_indeces_2 = np.where(np.diff(np.sign(mi_zc_roi_2 - 0.5)))[0]
        zero_indeces_3 = np.where(np.diff(np.sign(mi_zc_roi_3 - 0.5)))[0]
        t_zc_zeros_1 = t_zc_roi_1[zero_indeces_1]
        t_zc_zeros_2 = t_zc_roi_2[zero_indeces_2]
        t_zc_zeros_3 = t_zc_roi_3[zero_indeces_3]

        # Find zero points
        zero_indeces_1 = np.where(np.diff(np.sign(mi_zc_roi_1 - 0.5)))[0]
        t_zc_zeros_1 = t_zc_roi_1[zero_indeces_1]
        zero_indeces_2 = np.where(np.diff(np.sign(mi_zc_roi_2 - 0.5)))[0]
        t_zc_zeros_2 = t_zc_roi_2[zero_indeces_2]
        zero_indeces_3 = np.where(np.diff(np.sign(mi_zc_roi_3 - 0.5)))[0]
        t_zc_zeros_3 = t_zc_roi_3[zero_indeces_3]

        # Filter double crossing points
        dist_1 = []
        dist_2 = []
        dist_3 = []
        for i in range(len(t_zc_zeros_1)-1):
            dist_1.append(t_zc_zeros_1[i+1]-t_zc_zeros_1[i])
        for i in range(len(t_zc_zeros_2)-1):
            dist_2.append(t_zc_zeros_2[i+1]-t_zc_zeros_2[i])
        for i in range(len(t_zc_zeros_3)-1):
            dist_3.append(t_zc_zeros_3[i+1]-t_zc_zeros_3[i])

        indices_less_than_1e_4 = [i for i, d in enumerate(dist_1) if d < 1e-4]
        t_zc_zeros_1 = np.delete(t_zc_zeros_1, indices_less_than_1e_4)
        indices_less_than_1e_4 = [i for i, d in enumerate(dist_2) if d < 1e-4]
        t_zc_zeros_2 = np.delete(t_zc_zeros_2, indices_less_than_1e_4)
        indices_less_than_1e_4 = [i for i, d in enumerate(dist_3) if d < 1e-4]
        t_zc_zeros_3 = np.delete(t_zc_zeros_3, indices_less_than_1e_4)

        # Time to position conversion by linear interpolating the zero crossing points
        WL = self.WL
        wl_ref_1 = np.arange(len(t_zc_zeros_1))
        wl_ref_2 = np.arange(len(t_zc_zeros_2))
        wl_ref_3 = np.arange(len(t_zc_zeros_3))
        x_zc_roi_1 = WL/4 * np.interp(t_zc_roi_1, t_zc_zeros_1, wl_ref_1)
        x_zc_roi_2 = WL/4 * np.interp(t_zc_roi_2, t_zc_zeros_2, wl_ref_2)
        x_zc_roi_3 = WL/4 * np.interp(t_zc_roi_3, t_zc_zeros_3, wl_ref_3)

        # Identify flat edges on both sides
        edge_tolerance = 1e-6  # Define a tolerance for flatness
        deriv_1 = np.gradient(x_zc_roi_1)
        deriv_2 = np.gradient(x_zc_roi_2)
        deriv_3 = np.gradient(x_zc_roi_3)
        
        # Find flat regions at the beginning
        start_flat_1 = np.argmax(deriv_1 > edge_tolerance)
        start_flat_2 = np.argmax(deriv_2 > edge_tolerance)
        start_flat_3 = np.argmax(deriv_3 > edge_tolerance)
        
        # Find flat regions at the end
        end_flat_1 = len(deriv_1) - np.argmax(deriv_1[::-1] > edge_tolerance)
        end_flat_2 = len(deriv_2) - np.argmax(deriv_2[::-1] > edge_tolerance)
        end_flat_3 = len(deriv_3) - np.argmax(deriv_3[::-1] > edge_tolerance)

        # Remove flat edges
        if start_flat_1 < end_flat_1:
            x_zc_roi_1 = x_zc_roi_1[start_flat_1:end_flat_1]
        if start_flat_2 < end_flat_2:
            x_zc_roi_2 = x_zc_roi_2[start_flat_2:end_flat_2]
        if start_flat_3 < end_flat_3:
            x_zc_roi_3 = x_zc_roi_3[start_flat_3:end_flat_3]

        # SNOM signal normalization
        snom_zc_roi_1 = self.snom[mask_zc_1]
        snom_zc_roi_2 = self.snom[mask_zc_2]
        snom_zc_roi_3 = self.snom[mask_zc_3]
        snom_zc_roi_1 = (snom_zc_roi_1 - np.min(snom_zc_roi_1)) / (np.max(snom_zc_roi_1) - np.min(snom_zc_roi_1))
        snom_zc_roi_2 = (snom_zc_roi_2 - np.min(snom_zc_roi_2)) / (np.max(snom_zc_roi_2) - np.min(snom_zc_roi_2))
        snom_zc_roi_3 = (snom_zc_roi_3 - np.min(snom_zc_roi_3)) / (np.max(snom_zc_roi_3) - np.min(snom_zc_roi_3))
        snom_zc_roi_1 = snom_zc_roi_1[start_flat_1:end_flat_1]
        snom_zc_roi_2 = snom_zc_roi_2[start_flat_2:end_flat_2]
        snom_zc_roi_3 = snom_zc_roi_3[start_flat_3:end_flat_3]

        return x_zc_roi_1, snom_zc_roi_1, x_zc_roi_2, snom_zc_roi_2, x_zc_roi_3, snom_zc_roi_3

    def zero_crossing_interpol_sim(self, t, mi, snom):
        t_zc_roi, mi_zc_roi = t, mi
        mi_zc_roi = (mi_zc_roi - np.min(mi_zc_roi)) / (np.max(mi_zc_roi) - np.min(mi_zc_roi))

        # Find zero points
        zero_indeces = np.where(np.diff(np.sign(mi_zc_roi - 0.5)))[0]
        t_zc_zeros = t_zc_roi[zero_indeces]

        # Filter double crossing points
        dist_2 = []
        for i in range(len(t_zc_zeros)-1):
            dist_2.append(t_zc_zeros[i+1]-t_zc_zeros[i])
        

        indices_less_than_1e_4 = [i for i, d in enumerate(dist_2) if d < 1e-4]

        t_zc_zeros = np.delete(t_zc_zeros, indices_less_than_1e_4)

        # Time to position conversion by linear interpolating the zero crossing points
        WL = self.WL
        wl_ref = np.arange(len(t_zc_zeros))
        x_zc_roi = WL/4 * np.interp(t_zc_roi, t_zc_zeros, wl_ref)

        # Identify flat edges on both sides
        edge_tolerance = 1e-6  # Define a tolerance for flatness
        deriv = np.gradient(x_zc_roi)
        
        # Find flat regions at the beginning
        start_flat = np.argmax(deriv > edge_tolerance)
        
        # Find flat regions at the end
        end_flat = len(deriv) - np.argmax(deriv[::-1] > edge_tolerance)

        # Remove flat edges
        if start_flat < end_flat:
            x_zc_roi = x_zc_roi[start_flat:end_flat]
            

        # SNOM signal normalization
        snom_zc_roi = snom
        snom_zc_roi = (snom_zc_roi - np.min(snom_zc_roi)) / (np.max(snom_zc_roi) - np.min(snom_zc_roi))
        snom_zc_roi = snom_zc_roi[start_flat:end_flat]

        return x_zc_roi, snom_zc_roi

    # def EnvSin_fit(self, envelop_type, method):
    #     # --------- helpers ---------
    #     def _freq_guess(x, y):
    #         # FFT-based spatial frequency guess (resample to uniform grid)
    #         n = min(4096, max(512, 2**int(np.ceil(np.log2(len(x))))))
    #         xu = np.linspace(x.min(), x.max(), n)
    #         yu = np.interp(xu, x, y - np.median(y))
    #         dx = np.median(np.diff(xu))
    #         Y = np.fft.rfft(yu); f = np.fft.rfftfreq(n, d=dx)
    #         k = 1 + np.argmax(np.abs(Y[1:])) if len(f) > 2 else 1
    #         return max(float(f[k]), 1e-12)  # cycles per x-unit

    #     # --------- models: A(x) * (sin(2π k x + φ0) + c) ---------
    #     def make_model_gaussian():
    #         # params: [A0, mu, sigma, k, phi0, c]
    #         def model(x, A0, mu, sigma, k, phi0, c):
    #             x = np.asarray(x, float)
    #             env = A0 * np.exp(-0.5 * ((x - mu) / np.abs(sigma))**2)
    #             return env * (np.sin(2*np.pi*k*x + phi0) + c)
    #         return model

    #     def make_model_poly(deg=2):
    #         # params: [A0, a1..adeg, k, phi0, c], with x centered at x0
    #         def model(x, *p):
    #             x = np.asarray(x, float); x0 = x.mean()
    #             A0 = p[0]
    #             a  = np.asarray(p[1:1+deg]) if deg>0 else np.array([])
    #             k  = p[1+deg]; phi0 = p[2+deg]; c = p[3+deg]
    #             tt = x - x0
    #             env = A0 * (1.0 + (np.vander(tt, N=deg+1, increasing=True)[:,1:] @ a if deg>0 else 0.0))
    #             # Optional safety to avoid negative envelope:
    #             # env = np.maximum(env, 1e-12)
    #             return env * (np.sin(2*np.pi*k*x + phi0) + c)
    #         return model

    #     # --------- fitter ---------
    #     def fit_A_times_sin_plus_c(x, y, envelope="gaussian", deg=2, maxfev=50000):
    #         x = np.asarray(x, float).ravel()
    #         y = np.asarray(y, float).ravel()
    #         A0 = 0.5*(np.percentile(y,95) - np.percentile(y,5))
    #         k0 = _freq_guess(x, y)
    #         phi0 = 0.0
    #         c0 = 0.0  # you can try np.clip(np.mean(y)/max(A0,1e-12), -0.9, 0.9)

    #         if envelope == "gaussian":
    #             model = make_model_gaussian()
    #             mu0 = float(x[np.argmax(y)]) if np.isfinite(y).all() else x.mean()
    #             sigma0 = 0.25*(x.max()-x.min())
    #             p0 = [A0, mu0, sigma0, k0, phi0, c0]
    #             lb = [0.0, x.min(), 1e-12, 1e-12, -np.inf, -1.0]
    #             ub = [np.inf, x.max(),  np.inf,  np.inf,  np.inf,  1.0]
    #         else:
    #             model = make_model_poly(deg=deg)
    #             p0 = [A0] + [0.0]*deg + [k0, phi0, c0]
    #             lb = [0.0] + [-np.inf]*deg + [1e-12, -np.inf, -1.0]
    #             ub = [np.inf] + [ np.inf]*deg + [ np.inf,  np.inf,  1.0]

    #         popt, pcov = curve_fit(model, x, y, p0=p0, bounds=(lb, ub), maxfev=maxfev)
    #         yhat = model(x, *popt)
    #         resid = y - yhat
    #         rmse = float(np.sqrt(np.mean(resid**2)))
    #         return model, popt, pcov, yhat, resid, rmse

        
    #     if method == 0:
    #         x_data = self.x_linear
    #         y_data = self.snom_linear
    #     if method == 1:
    #         x_data = self.x_lin_interpol
    #         y_data = self.snom_lin_interpol
    #     if method == 2:
    #         x_data = self.x_nonlinear
    #         y_data = self.snom_nonlinear

    #     dof = 10
    #     # Choose ONE:
    #     if envelop_type == 0:
    #         model, popt, pcov, yhat, resid, rmse = fit_A_times_sin_plus_c(x_data, y_data, envelope="gaussian")
    #     # or:
    #     if envelop_type == 1:
    #         model, popt, pcov, yhat, resid, rmse = fit_A_times_sin_plus_c(x_data, y_data, envelope="poly", deg=dof)
            
    #     xx = np.linspace(x_data.min(), x_data.max(), 2000)
    #     yy = model(xx, *popt)

    #     # Plot
    #     if 1:
    #         fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7,6), sharex=True, gridspec_kw={'height_ratios':[3,1.2]})
    #         ax1.plot(x_data, y_data, '.', ms=3, label='Data')
    #         ax1.plot(xx, yy, '-', lw=0.8, label='Fit', color='k')
    #         ax1.set_ylabel('Norm. SNOM signal'); ax1.legend()
    #         ax1.set_title('Fit: A(x) * (sin(2π k x + φ0) + c)')
    #         ax2.plot(x_data, y_data - yhat, '.', ms=3, color='C3', label='Residuals')
    #         ax2.axhline(0, color='k', lw=0.8, ls='--'); ax2.set_xlabel('Position (um)'); ax2.set_ylabel('Residual'); ax2.legend()
    #         plt.tight_layout(); plt.show()
    #     # print('RMSE:', rmse); print('popt:', popt); print('Propagation const:', popt[-3], ' /um')
    #     beta = popt[-3]*np.pi
    #     beta_error = np.sqrt(pcov[-3][-3])*np.pi
        
    #     # find residue
    #     residue = y_data - yhat


    #     if max(abs(residue)) < 100: # keep this 0.11
    #         return xx, yy, beta, beta_error
    #     else:
    #         return None, None, None, None
    

    def EnvSin_fit(self, envelop_type, method, k_bounds=[0.5, 10], maxfev=50000):
        # --------- helpers ---------
        def _freq_guess(x, y):
            n = min(4096, max(512, 2**int(np.ceil(np.log2(len(x))))))
            xu = np.linspace(x.min(), x.max(), n)
            yu = np.interp(xu, x, y - np.median(y))
            dx = np.median(np.diff(xu))
            Y = np.fft.rfft(yu); f = np.fft.rfftfreq(n, d=dx)
            k = 1 + np.argmax(np.abs(Y[1:])) if len(f) > 2 else 1
            return max(float(f[k]), 1e-12)  # cycles per x-unit

        # --------- models: A(x) * (sin(2π k x + φ0) + c) ---------
        def make_model_gaussian():
            # params: [A0, mu, sigma, k, phi0, c]
            def model(x, A0, mu, sigma, k, phi0, c):
                x = np.asarray(x, float)
                env = A0 * np.exp(-0.5 * ((x - mu) / np.abs(sigma))**2)
                return env * (np.sin(2*np.pi*k*x + phi0) + c)
            return model

        def make_model_poly(deg=2):
            # params: [A0, a1..adeg, k, phi0, c] with x centered
            def model(x, *p):
                x = np.asarray(x, float); x0 = x.mean()
                A0 = p[0]
                a  = np.asarray(p[1:1+deg]) if deg>0 else np.array([])
                k  = p[1+deg]; phi0 = p[2+deg]; c = p[3+deg]
                tt = x - x0
                env = A0 * (1.0 + (np.vander(tt, N=deg+1, increasing=True)[:,1:] @ a if deg>0 else 0.0))
                return env * (np.sin(2*np.pi*k*x + phi0) + c)
            return model

        # --------- fitter with k-bounds ---------
        def fit_A_times_sin_plus_c(x, y, envelope="gaussian", deg=2, maxfev=50000, k_bounds=None):
            x = np.asarray(x, float).ravel()
            y = np.asarray(y, float).ravel()
            A0 = 0.5*(np.percentile(y,95) - np.percentile(y,5))
            k0 = _freq_guess(x, y)
            phi0 = 0.0
            c0 = 0.0

            if envelope == "gaussian":
                model = make_model_gaussian()
                mu0 = float(x[np.argmax(y)]) if np.isfinite(y).all() else x.mean()
                sigma0 = 0.25*(x.max()-x.min())
                p0 = [A0, mu0, sigma0, k0,  phi0, c0]
                lb = [0.0, x.min(), 1e-12, 1e-12, -np.inf, -1.0]
                ub = [np.inf, x.max(),  np.inf,  np.inf,  np.inf,  1.0]
                idx_k = 3
            else:
                model = make_model_poly(deg=deg)
                p0 = [A0] + [0.0]*deg + [k0, phi0, c0]
                lb = [0.0] + [-np.inf]*deg + [1e-12, -np.inf, -1.0]
                ub = [np.inf] + [ np.inf]*deg + [ np.inf,  np.inf,  1.0]
                idx_k = 1 + deg  # k position

            # ---- apply user-provided k bounds, if any ----
            if k_bounds is not None:
                k_lo, k_hi = map(float, k_bounds)
                # ensure valid
                if not (np.isfinite(k_lo) and np.isfinite(k_hi) and k_lo < k_hi):
                    raise ValueError("k_bounds must be (k_min, k_max) with k_min < k_max.")
                lb[idx_k] = max(lb[idx_k], k_lo)
                ub[idx_k] = min(ub[idx_k], k_hi)
                # keep initial guess inside bounds
                p0[idx_k] = float(np.clip(p0[idx_k], lb[idx_k], ub[idx_k]))

            popt, pcov = curve_fit(model, x, y, p0=p0, bounds=(lb, ub), maxfev=maxfev)
            yhat = model(x, *popt)
            resid = y - yhat
            rmse = float(np.sqrt(np.mean(resid**2)))
            return model, popt, pcov, yhat, resid, rmse, idx_k

        # --------- choose data ---------
        if method == 0:
            x_data, y_data = self.x_linear, self.snom_linear
        elif method == 1:
            x_data, y_data = self.x_lin_interpol, self.snom_lin_interpol
        else:
            x_data, y_data = self.x_nonlinear, self.snom_nonlinear

        dof = 10
        if envelop_type == 0:
            model, popt, pcov, yhat, resid, rmse, idx_k = fit_A_times_sin_plus_c(
                x_data, y_data, envelope="gaussian", maxfev=maxfev, k_bounds=k_bounds
            )
        else:
            model, popt, pcov, yhat, resid, rmse, idx_k = fit_A_times_sin_plus_c(
                x_data, y_data, envelope="poly", deg=dof, maxfev=maxfev, k_bounds=k_bounds
            )

        xx = np.linspace(x_data.min(), x_data.max(), 2000)
        yy = model(xx, *popt)

        # k is popt[idx_k]; with sin(2π k x + φ0), the propagation constant β = 2π k
        k_fit = float(popt[idx_k])
        beta = 2*np.pi*k_fit
        beta_error = 2*np.pi*np.sqrt(max(pcov[idx_k, idx_k], 0.0))

        residue = y_data - yhat

        # Plot
        if 0:
            fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7,6), sharex=True, gridspec_kw={'height_ratios':[3,1.2]})
            ax1.plot(x_data, y_data, '.', ms=3, label='Data')
            ax1.plot(xx, yy, '-', lw=0.8, label='Fit', color='k')
            ax1.set_ylabel('Norm. SNOM signal'); ax1.legend()
            ax1.set_title('Fit: A(x) * (sin(2π k x + φ0) + c)')
            ax2.plot(x_data, y_data - yhat, '.', ms=3, color='C3', label='Residuals')
            ax2.axhline(0, color='k', lw=0.8, ls='--'); ax2.set_xlabel('Position (um)'); ax2.set_ylabel('Residual'); ax2.legend()
            plt.tight_layout(); plt.show()

        if np.max(np.abs(residue)) < 0.6:
            return xx, yy, beta, beta_error
        else:
            return None, None, None, None

    def EnvSin_fit_section_by_section(self, envelop_type, xx, yy):
        # --------- helpers ---------
        def _freq_guess(x, y):
            # FFT-based spatial frequency guess (resample to uniform grid)
            n = min(4096, max(512, 2**int(np.ceil(np.log2(len(x))))))
            xu = np.linspace(x.min(), x.max(), n)
            yu = np.interp(xu, x, y - np.median(y))
            dx = np.median(np.diff(xu))
            Y = np.fft.rfft(yu); f = np.fft.rfftfreq(n, d=dx)
            k = 1 + np.argmax(np.abs(Y[1:])) if len(f) > 2 else 1
            return max(float(f[k]), 1e-12)  # cycles per x-unit

        # --------- models: A(x) * (sin(2π k x + φ0) + c) ---------
        def make_model_gaussian():
            # params: [A0, mu, sigma, k, phi0, c]
            def model(x, A0, mu, sigma, k, phi0, c):
                x = np.asarray(x, float)
                env = A0 * np.exp(-0.5 * ((x - mu) / np.abs(sigma))**2)
                return env * (np.sin(2*np.pi*k*x + phi0) + c)
            return model

        def make_model_poly(deg=2):
            # params: [A0, a1..adeg, k, phi0, c], with x centered at x0
            def model(x, *p):
                x = np.asarray(x, float); x0 = x.mean()
                A0 = p[0]
                a  = np.asarray(p[1:1+deg]) if deg>0 else np.array([])
                k  = p[1+deg]; phi0 = p[2+deg]; c = p[3+deg]
                tt = x - x0
                env = A0 * (1.0 + (np.vander(tt, N=deg+1, increasing=True)[:,1:] @ a if deg>0 else 0.0))
                # Optional safety to avoid negative envelope:
                # env = np.maximum(env, 1e-12)
                return env * (np.sin(2*np.pi*k*x + phi0) + c)
            return model

        # --------- fitter ---------
        def fit_A_times_sin_plus_c(x, y, envelope="gaussian", deg=2, maxfev=50000):
            x = np.asarray(x, float).ravel()
            y = np.asarray(y, float).ravel()
            A0 = 0.5*(np.percentile(y,95) - np.percentile(y,5))
            k0 = _freq_guess(x, y)
            phi0 = 0.0
            c0 = 0.0  # you can try np.clip(np.mean(y)/max(A0,1e-12), -0.9, 0.9)

            if envelope == "gaussian":
                model = make_model_gaussian()
                mu0 = float(x[np.argmax(y)]) if np.isfinite(y).all() else x.mean()
                sigma0 = 0.25*(x.max()-x.min())
                p0 = [A0, mu0, sigma0, k0, phi0, c0]
                lb = [0.0, x.min(), 1e-12, 1e-12, -np.inf, -1.0]
                ub = [np.inf, x.max(),  np.inf,  np.inf,  np.inf,  1.0]
            else:
                model = make_model_poly(deg=deg)
                p0 = [A0] + [0.0]*deg + [k0, phi0, c0]
                lb = [0.0] + [-np.inf]*deg + [1e-12, -np.inf, -1.0]
                ub = [np.inf] + [ np.inf]*deg + [ np.inf,  np.inf,  1.0]

            popt, pcov = curve_fit(model, x, y, p0=p0, bounds=(lb, ub), maxfev=maxfev)
            yhat = model(x, *popt)
            resid = y - yhat
            rmse = float(np.sqrt(np.mean(resid**2)))
            return model, popt, pcov, yhat, resid, rmse

        x_data = xx
        y_data = yy

        dof = 5
        # Choose ONE:
        if envelop_type == 0:
            model, popt, pcov, yhat, resid, rmse = fit_A_times_sin_plus_c(x_data, y_data, envelope="gaussian")

        # or:
        if envelop_type == 1:
            model, popt, pcov, yhat, resid, rmse = fit_A_times_sin_plus_c(x_data, y_data, envelope="poly", deg=dof)

            
        xx_ = np.linspace(x_data.min(), x_data.max(), 2000)
        yy_ = model(xx_, *popt)


        # Plot
        if 1:
            fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7,6), sharex=True, gridspec_kw={'height_ratios':[3,1.2]})
            ax1.plot(x_data, y_data, '.', ms=3, label='Data')
            ax1.plot(xx_, yy_, '-', lw=0.8, label='Fit', color='k')
            ax1.set_ylabel('Norm. SNOM signal'); ax1.legend()
            ax1.set_title('Fit: A(x) * (sin(2π k x + φ0) + c)')
            ax2.plot(x_data, y_data - yhat, '.', ms=3, color='C3', label='Residuals')
            ax2.axhline(0, color='k', lw=0.8, ls='--'); ax2.set_xlabel('Position (um)'); ax2.set_ylabel('Residual'); ax2.legend()
            plt.tight_layout(); plt.show()

            plt.tight_layout(); plt.show()

        # print('RMSE:', rmse); print('popt:', popt); print('Propagation const:', popt[-3], ' /um')
        beta = popt[-3]*np.pi

        beta_error = np.sqrt(pcov[-3][-3])*np.pi


        # find residue
        residue = y_data - yhat


        if max(abs(residue)) < 0.11:
            return xx, yy, beta, beta_error
        else:
            return None, None, None, None
        
    def FFT_with_Gaussian_fit(self,x_mi,y_mi, x_snom, y_snom):

        # === Load calibration file ===
        data = np.load("./pitches_and_diameters_1389.npz")
        pitch_values = data["pitches_1389"]
        interpolated_dia_values = data["diameters_1389"]

        # load wavelength
        WL = 1389 # nm

        # =========================
        # Zero-padding FFT
        # =========================
        def single_sided_fft(t, y, min_freq_hz=0.0, n_fft=None):
            """
            Compute single-sided amplitude spectrum (windowed + amplitude-corrected),
            with optional zero-padding to n_fft length.

            Returns:
                freqs (Hz), amps, f_peak (Hz), a_peak
            """
            # Clean NaNs/Infs and ensure 1D
            t = np.asarray(t, float).ravel()
            y = np.asarray(y, float).ravel()
            finite = np.isfinite(t) & np.isfinite(y)
            t, y = t[finite], y[finite]

            if t.size < 2:
                raise ValueError("Not enough samples in ROI for FFT.")

            # Sample interval / rate (assume nearly uniform)
            dt = np.median(np.diff(t))
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError("Time vector isn't monotonic or has invalid spacing.")
            fs = 1.0 / dt
            N = y.size

            # Detrend (remove mean) and window to reduce leakage
            y = y - np.mean(y)
            w = np.hanning(N)
            cg = w.mean()  # coherent gain for amplitude correction
            yw = y * w

            # ---- zero padding length ----
            if n_fft is None:
                n_fft = N
            elif n_fft < N:
                raise ValueError("n_fft must be >= len(y) for zero-padding.")

            # rFFT (with possible zero padding)
            Y = np.fft.rfft(yw, n=n_fft)
            freqs = np.fft.rfftfreq(n_fft, d=dt)

            # Single-sided amplitude spectrum (correct for window + single-sided scaling)
            amps = (2.0 / (N * cg)) * np.abs(Y)

            # Fix DC and Nyquist scaling (DC shouldn't be doubled; Nyquist exists if n_fft even)
            if amps.size > 0:
                amps[0] *= 0.5
            if (n_fft % 2 == 0) and (amps.size > 1):
                amps[-1] *= 0.5

            # Find peak excluding DC and below min_freq_hz
            idx_valid = np.where(freqs >= max(min_freq_hz, 0.0))[0]
            if idx_valid.size == 0:
                raise ValueError("No valid frequency bins after applying min_freq_hz.")

            # Exclude the first bin if it's DC (freq==0)
            if freqs[idx_valid[0]] == 0.0 and idx_valid.size > 1:
                idx_valid = idx_valid[1:]

            if idx_valid.size == 0:
                raise ValueError("Only DC present; cannot determine peak frequency.")

            i_peak_local = idx_valid[np.argmax(amps[idx_valid])]
            f_peak = float(freqs[i_peak_local])
            a_peak = float(amps[i_peak_local])
            return freqs, amps, f_peak, a_peak

        # Define a Gaussian function
        def gaussian(x, a, x0, sigma):
            return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

        def nearest_diameter_from_pitch(pitches_m, pitch_values_m, dia_vals):
            """
            For each measured pitch in meters, find j that minimizes |pitch_values_m[j] - pitches_m[i]|
            and return dia_vals[j]. Works for 1D arrays.
            """
            pitches_m = np.asarray(pitches_m, float).ravel()
            pv = np.asarray(pitch_values_m, float).ravel()
            dv = np.asarray(dia_vals, float).ravel()
            idx = np.abs(pitches_m[:, None] - pv[None, :]).argmin(axis=1)
            return dv[idx], idx

        # =========================
        # FFT with zero-padding
        # =========================
        PAD = 5                # <-- set your zero-padding factor here
        MIN_FREQ_HZ = 1.0        # ignore DC/very slow drift when finding peak

        # SNOM raw
        freqs1, amps1, f_peak1, a_peak1 = single_sided_fft(
            x_snom, y_snom, min_freq_hz=MIN_FREQ_HZ, n_fft=PAD*len(y_snom)
        )

        # MI raw
        freqs2, amps2, f_peak2, a_peak2 = single_sided_fft(
            x_mi, y_mi, min_freq_hz=MIN_FREQ_HZ, n_fft=PAD*len(y_mi)
        )

        if 0:
            plt.figure()
            plt.plot(freqs1, amps1, 'o', label='SNOM')
            plt.plot(freqs2, amps2, 'o', label='MI')
            plt.xlabel('Freq (Hz)'); plt.ylabel('a.u.')
            plt.xlim([f_peak1 - 40, f_peak1 + 40]); plt.grid(True); plt.legend(); plt.show()

        # =========================
        # Gaussian peak fit (fixed the MI arrays)
        # =========================
        window_size = 100

        if f_peak1 > 20:
            # SNOM window around peak
            mask_peak_snom = (freqs1 >= f_peak1 - window_size) & (freqs1 <= f_peak1 + window_size)
            freqs_peak_snom = freqs1[mask_peak_snom]
            amps_peak_snom  = amps1[mask_peak_snom]
            popt, _ = curve_fit(gaussian, freqs_peak_snom, amps_peak_snom, p0=[amps_peak_snom.max(), f_peak1, max(1.0, window_size/10)])
            a_gauss_s, f_peak_gauss_s, sigma_gauss_s = popt
            freqs_fit_snom = np.linspace(freqs_peak_snom.min(), freqs_peak_snom.max(), 100000)
            amps_fit_snom  = gaussian(freqs_fit_snom, a_gauss_s, f_peak_gauss_s, sigma_gauss_s)
            Pfreq_snom_fit = f_peak_gauss_s

            # MI window around peak  (FIX: use freqs2/amps2 here)
            mask_peak_mi = (freqs2 >= f_peak2 - window_size) & (freqs2 <= f_peak2 + window_size)
            freqs_peak_mi = freqs2[mask_peak_mi]
            amps_peak_mi  = amps2[mask_peak_mi]
            popt, _ = curve_fit(gaussian, freqs_peak_mi, amps_peak_mi, p0=[amps_peak_mi.max(), f_peak2, max(1.0, window_size/10)])
            a_gauss_m, f_peak_gauss_m, sigma_gauss_m = popt
            freqs_fit_mi = np.linspace(freqs_peak_mi.min(), freqs_peak_mi.max(), 1000)
            amps_fit_mi  = gaussian(freqs_fit_mi, a_gauss_m, f_peak_gauss_m, sigma_gauss_m)
            Pfreq_mi_fit = f_peak_gauss_m

        if f_peak1 <= 20:
            # SNOM window around peak
            mask_peak_snom = (freqs1 >= f_peak2 - window_size) & (freqs1 <= f_peak2 + window_size)
            freqs_peak_snom = freqs1[mask_peak_snom]
            amps_peak_snom  = amps1[mask_peak_snom]
            popt, _ = curve_fit(gaussian, freqs_peak_snom, amps_peak_snom, p0=[amps_peak_snom.max(), f_peak2, max(1.0, window_size/10)])
            a_gauss_s, f_peak_gauss_s, sigma_gauss_s = popt
            freqs_fit_snom = np.linspace(freqs_peak_snom.min(), freqs_peak_snom.max(), 100000)
            amps_fit_snom  = gaussian(freqs_fit_snom, a_gauss_s, f_peak_gauss_s, sigma_gauss_s)
            Pfreq_snom_fit = f_peak_gauss_s

            # MI window around peak  (FIX: use freqs2/amps2 here)
            mask_peak_mi = (freqs2 >= f_peak2 - window_size) & (freqs2 <= f_peak2 + window_size)
            freqs_peak_mi = freqs2[mask_peak_mi]
            amps_peak_mi  = amps2[mask_peak_mi]
            popt, _ = curve_fit(gaussian, freqs_peak_mi, amps_peak_mi, p0=[amps_peak_mi.max(), f_peak2, max(1.0, window_size/10)])
            a_gauss_m, f_peak_gauss_m, sigma_gauss_m = popt
            freqs_fit_mi = np.linspace(freqs_peak_mi.min(), freqs_peak_mi.max(), 1000)
            amps_fit_mi  = gaussian(freqs_fit_mi, a_gauss_m, f_peak_gauss_m, sigma_gauss_m)
            Pfreq_mi_fit = f_peak_gauss_m
        
        # Optional plots
        if 0:
            plt.figure()
            plt.plot(freqs1, amps1, 'o', label='SNOM spectrum')
            plt.plot(freqs_fit_snom, amps_fit_snom, '-', color='black')
            plt.xlabel('Freq (Hz)'); plt.ylabel('a.u.')
            plt.xlim([f_peak1 - window_size, f_peak1 + window_size]); plt.grid(True); plt.legend()

            plt.plot(freqs2, amps2, 'o', label='MI spectrum')
            plt.plot(freqs_fit_mi, amps_fit_mi, '--', color='black' )
            plt.xlabel('Freq (Hz)'); plt.ylabel('a.u.')
            plt.xlim([f_peak2 - window_size, f_peak2 + window_size]); plt.grid(True); plt.legend()
            plt.show()

        # =========================
        # Pitch + diameter (assumes WL, pitch_values, interpolated_dia_values are defined)
        # =========================
        # Example: if your wavelength WL is in nm and you want pitch in mm, adjust units accordingly.
        # Here, follow your original formulae:
        n_eff_raw = Pfreq_snom_fit / Pfreq_mi_fit
        beta_raw = 2*np.pi*n_eff_raw / WL
        pitch_raw = np.pi / beta_raw * 1e-3    # <- your original unit scaling

        n_eff = f_peak1 / f_peak2
        beta_ = 2*np.pi*n_eff / WL
        pitch_ = np.pi / beta_ * 1e-3          # <- your original unit scaling

        dia_raw, index = nearest_diameter_from_pitch(pitch_raw, pitch_values, interpolated_dia_values) # fit
        dia_,   ind   = nearest_diameter_from_pitch(pitch_,    pitch_values, interpolated_dia_values) # take max

        print('Fit Gaussing diameter with zero-padding: ',dia_raw[0]*1e9, 'nm' )
        print('Diameter with zero-padding: ',dia_[0]*1e9, 'nm' )
        
        return dia_raw, dia_
    
    def FFT_with_skewed_Gaussian_fit(self, x_mi, y_mi, x_snom, y_snom):
        """
        Perform FFT-based skewed Gaussian fitting on MI and SNOM spectra.
        - Fits unpadded FFT (for accuracy)
        - Plots padded FFT (for smooth appearance)
        - Evaluates fit on same frequency bins (no 'shorter' curve effect)
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit
        from scipy.special import erf

        # === Load calibration file ===
        data = np.load("./pitches_and_diameters_1389.npz")
        pitch_values = data["pitches_1389"]
        interpolated_dia_values = data["diameters_1389"]
        WL = 1389  # nm

        # === Helper functions ===
        def single_sided_fft(t, y, min_freq_hz=0.0, n_fft=None):
            t, y = np.asarray(t, float).ravel(), np.asarray(y, float).ravel()
            finite = np.isfinite(t) & np.isfinite(y)
            t, y = t[finite], y[finite]
            dt = np.median(np.diff(t))
            N = len(y)
            if n_fft is None: n_fft = N
            w = np.hanning(N)
            Y = np.fft.rfft((y - y.mean()) * w, n=n_fft)
            freqs = np.fft.rfftfreq(n_fft, d=dt)
            amps = 2 * np.abs(Y) / (N * w.mean())
            if len(amps) > 1: amps[-1] *= 0.5
            idx = freqs >= min_freq_hz
            freqs, amps = freqs[idx], amps[idx]
            i_peak = np.argmax(amps)
            return freqs, amps, freqs[i_peak], amps[i_peak]

        def parabolic_refine(x, y, i):
            if i <= 0 or i >= len(y) - 1:
                return x[i], y[i]
            y1, y2, y3 = y[i-1], y[i], y[i+1]
            denom = y1 - 2*y2 + y3
            if denom == 0: return x[i], y2
            delta = 0.5 * (y1 - y3) / denom
            dx = x[1] - x[0]
            x_peak = x[i] + delta * dx
            y_peak = y2 - 0.25 * (y1 - y3) * delta
            return x_peak, y_peak

        def skewed_gaussian(x, a, x0, sigma, skew, baseline):
            z = (x - x0) / (sigma + 1e-20)
            return a * np.exp(-0.5*z*z) * (1 + erf(skew * z / np.sqrt(2))) + baseline

        def fit_skewed_peak_window(freqs, amps, f_center, window_hz=80.0, decimate=1):
            m = (freqs >= f_center - window_hz) & (freqs <= f_center + window_hz)
            fx, ay = freqs[m], amps[m]
            if len(fx) < 5:
                i = np.argmax(amps)
                f_par, _ = parabolic_refine(freqs, amps, i)
                return f_par, (None, None), (fx, ay, ay.copy())
            fx_fit, ay_fit = fx[::decimate], ay[::decimate]
            baseline0, a0, x0 = np.median(ay_fit), ay_fit.max()-np.median(ay_fit), fx_fit[np.argmax(ay_fit)]
            bin_w = np.median(np.diff(fx_fit))
            sigma0 = max((4*bin_w)/(2*np.sqrt(2*np.log(2))), bin_w)
            p0 = [a0, x0, sigma0, 0.0, baseline0]
            lb = [0.0, f_center - 2*window_hz, bin_w, -6.0, 0.0]
            ub = [np.inf, f_center + 2*window_hz, 0.5*window_hz, 6.0, np.inf]
            popt, pcov = curve_fit(skewed_gaussian, fx_fit, ay_fit, p0=p0, bounds=(lb, ub), maxfev=20000)
            ay_model = skewed_gaussian(fx, *popt)
            return float(popt[1]), (popt, pcov), (fx, ay, ay_model)

        def nearest_diameter_from_pitch(pitches_m, pitch_values_m, dia_vals):
            pitches_m = np.asarray(pitches_m, float).ravel()
            pv, dv = np.asarray(pitch_values_m, float).ravel(), np.asarray(dia_vals, float).ravel()
            idx = np.abs(pitches_m[:, None] - pv[None, :]).argmin(axis=1)
            return dv[idx], idx

        # === Parameters ===
        PAD_PLOT = 10
        WINDOW_HZ = 80
        DECIMATE = 2
        MIN_FREQ_HZ = 1.0

        # === FFTs ===
        freqs_mi, amps_mi, fpk_mi, _ = single_sided_fft(x_mi, y_mi, min_freq_hz=MIN_FREQ_HZ)
        freqs_snom, amps_snom, fpk_snom, _ = single_sided_fft(x_snom, y_snom, min_freq_hz=MIN_FREQ_HZ)
        freqs_mi_pad, amps_mi_pad, fpk_mi_pad, _ = single_sided_fft(x_mi, y_mi, min_freq_hz=MIN_FREQ_HZ, n_fft=PAD_PLOT*len(y_mi))
        freqs_snom_pad, amps_snom_pad, fpk_snom_pad, _ = single_sided_fft(x_snom, y_snom, min_freq_hz=MIN_FREQ_HZ, n_fft=PAD_PLOT*len(y_snom))

        use_pad = True
        if use_pad:
            # === Fit MI first ===
            f_mi_fit, (popt_mi, _), (fx_mi, ay_mi, ay_mi_fit) = fit_skewed_peak_window(freqs_mi_pad, amps_mi_pad, fpk_mi_pad, window_hz=WINDOW_HZ, decimate=DECIMATE)

            # === Fit SNOM anchored on MI ===
            f_snom_fit, (popt_snom, _), (fx_sn, ay_sn, ay_sn_fit) = fit_skewed_peak_window(freqs_snom_pad, amps_snom_pad, fpk_snom_pad, window_hz=WINDOW_HZ, decimate=DECIMATE)
        else:
            # === Fit MI first ===
            f_mi_fit, (popt_mi, _), (fx_mi, ay_mi, ay_mi_fit) = fit_skewed_peak_window(freqs_mi, amps_mi, fpk_mi, window_hz=WINDOW_HZ, decimate=DECIMATE)

            # === Fit SNOM anchored on MI ===
            f_snom_fit, (popt_snom, _), (fx_sn, ay_sn, ay_sn_fit) = fit_skewed_peak_window(freqs_snom, amps_snom, f_mi_fit, window_hz=WINDOW_HZ, decimate=DECIMATE)

        # === Derived quantities ===
        n_eff = f_snom_fit / f_mi_fit
        # n_eff = fpk_snom_pad / fpk_mi_pad
        beta = 2*np.pi*n_eff / WL
        pitch_m = np.pi / beta * 1e-3
        dia_fit, _ = nearest_diameter_from_pitch(pitch_m, pitch_values, interpolated_dia_values)
        print(f"MI={f_mi_fit:.6f} Hz | SNOM={f_snom_fit:.6f} Hz | n_eff={n_eff:.6f} |pitch = {pitch_m:.6f} um | Dia≈{dia_fit[0]*1e9:.2f} nm")
        # === Plot ===
        if 0:
            plt.figure(figsize=(7, 4.5))
            m1 = (freqs_mi_pad >= f_mi_fit - WINDOW_HZ) & (freqs_mi_pad <= f_mi_fit + WINDOW_HZ)
            m2 = (freqs_snom_pad >= f_mi_fit - WINDOW_HZ) & (freqs_snom_pad <= f_mi_fit + WINDOW_HZ)
            plt.plot(freqs_mi_pad[m1], amps_mi_pad[m1], 'o', ms=3, label='MI')
            plt.plot(freqs_mi_pad[m1], skewed_gaussian(freqs_mi_pad[m1], *popt_mi), '-', color='black', lw=1.2, label='MI skewed-G fit')
            plt.plot(freqs_snom_pad[m2], amps_snom_pad[m2], 'o', ms=3, label='SNOM')
            plt.plot(freqs_snom_pad[m2], skewed_gaussian(freqs_snom_pad[m2], *popt_snom), '--', color='black', lw=1.2, label='SNOM skewed-G fit')
            ctr = np.median([f_mi_fit, f_snom_fit])
            plt.xlim([ctr - 80, ctr + 80])
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude (a.u.)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

            # # === Residuals ===
            # plt.figure(figsize=(7, 3))
            # plt.plot(fx_mi, ay_mi - ay_mi_fit, '.', ms=3)
            # plt.axhline(0, color='k', lw=0.8, ls='--')
            # plt.title('MI Residuals')
            # plt.tight_layout()
            # plt.show()

            # plt.figure(figsize=(7, 3))
            # plt.plot(fx_sn, ay_sn - ay_sn_fit, '.', ms=3, color='C1')
            # plt.axhline(0, color='k', lw=0.8, ls='--')
            # plt.title('SNOM Residuals')
            # plt.tight_layout()
            # plt.show()

        return dia_fit
 
    def nonlinear_method_diameter(self, x_mi, y_mi, x_snom, y_snom, i = 0): 

        # === Load calibration file ===
        data = np.load("./pitches_and_diameters_1389.npz")
        pitch_values = data["pitches_1389"]
        interpolated_dia_values = data["diameters_1389"]

        # load wavelength
        WL = 1389 # nm
        def nearest_diameter_from_pitch(pitches_m, pitch_values_m, dia_vals):
            pitches_m = np.asarray(pitches_m, float).ravel()
            pv = np.asarray(pitch_values_m, float).ravel()
            dv = np.asarray(dia_vals, float).ravel()
            idx = np.abs(pitches_m[:, None] - pv[None, :]).argmin(axis=1)
            return dv[idx], idx
        
        # nonlinear fit to mi signal
        x_, y_ = self.nonlinear_method_sim(x_mi, y_mi)
        self.snom_nonlinear = (y_snom - np.min(y_snom)) / (np.max(y_snom) - np.min(y_snom))
        xx_, yy_, beta_nonlin, beta_nonlin_err = self.EnvSin_fit(i,2) # i = 0: Gaussian fit, i = 1: poly amplitude fit
        if beta_nonlin is not None:
            pitch_nonlin = 2*np.pi/beta_nonlin
            print('Nonlinear phase fit: ',pitch_nonlin, 'um')
            dia_nonlin, index = nearest_diameter_from_pitch(pitch_nonlin, pitch_values, interpolated_dia_values)
            print('Nonlinear phase fit and time to position conversion gives diameter: ',dia_nonlin[0]*1e9, 'nm' )
            return dia_nonlin
        else:
            return np.array([np.nan])


    def zero_crossing_interpol_method_diameter(self, x_mi, y_mi, y_snom, i = 0):
        # === Load calibration file ===
        data = np.load("./pitches_and_diameters_1389.npz")
        pitch_values = data["pitches_1389"]
        interpolated_dia_values = data["diameters_1389"]

        # load wavelength
        WL = 1389 # nm
        def nearest_diameter_from_pitch(pitches_m, pitch_values_m, dia_vals):
            pitches_m = np.asarray(pitches_m, float).ravel()
            pv = np.asarray(pitch_values_m, float).ravel()
            dv = np.asarray(dia_vals, float).ravel()
            idx = np.abs(pitches_m[:, None] - pv[None, :]).argmin(axis=1)
            return dv[idx], idx
        
        # nonlinear fit to mi signal
        x_, y_ = self.zero_crossing_interpol_sim(x_mi, y_mi, y_snom)
        self.x_lin_interpol = x_
        self.snom_lin_interpol = (y_ - np.min(y_)) / (np.max(y_) - np.min(y_))
        xx_, yy_, beta_zc, beta_zc_err = self.EnvSin_fit(i,1) # i = 0: Gaussian fit, i = 1: poly amplitude fit
        pitch_zc = np.pi/beta_zc
        dia_zc, index = nearest_diameter_from_pitch(pitch_zc, pitch_values, interpolated_dia_values)
        print('Zero-crossing and time to position conversion gives diameter: ',dia_zc[0]*1e9, 'nm' )
        return dia_zc  
    
    def weighted_cal(self, beta, beta_err):
        if len(beta) != len(beta_err):
            raise ValueError(f"beta and beta_err must have same length, got {len(beta)} vs {len(beta_err)}")

        beta = np.asarray(beta, float)
        err  = np.asarray(beta_err, float)
        m = np.isfinite(beta) & np.isfinite(err) & (err > 0)
        if not np.any(m):
            raise ValueError("No valid points for weighted mean.")
        beta = beta[m]; err = err[m]
        w = 1.0/(err**2)
        Mw = np.sum(w*beta)/np.sum(w)
        SMEw = np.sqrt(1.0/np.sum(w))
        return Mw, SMEw

    
    def diamter_calculation(self, pitch, pitch_err):

        solver = CompactFiberSolver(self.WL*1e-6, l=1)
        n_silica = solver.sellmeier_silica(self.WL)
        n_air = 1.0

        neff_target = (self.WL/2*1e-6)/pitch
        beta_target = neff_target * solver.k0

        d0 = solver.diameter_from_beta(beta_target, n_silica, n_air,
                                    a_min_um=0.005, a_max_um=1.3,  # 半径の探索範囲 [μm]
                           n_scan=400, max_iter=20, eps=1e-10)

        pitch_plus = pitch + pitch_err
        neff_target_plus = (self.WL/2*1e-6)/pitch_plus
        beta_target_plus = neff_target_plus * solver.k0

        pitch_minus = pitch - pitch_err
        neff_target_minus = (self.WL/2*1e-6)/pitch_minus
        beta_target_minus = neff_target_minus * solver.k0

        d_plus = solver.diameter_from_beta(beta_target_minus, n_silica, n_air,
                                    a_min_um=0.005, a_max_um=1.3,  # 半径の探索範囲 [μm]
                           n_scan=400, max_iter=20, eps=1e-10)

        d_minus = solver.diameter_from_beta(beta_target_plus, n_silica, n_air,
                                    a_min_um=0.005, a_max_um=1.3,  # 半径の探索範囲 [μm]
                           n_scan=400, max_iter=20, eps=1e-10)

        d = (d_plus + d_minus) / 2
        d_err = (d_plus - d_minus) / 2

        return d, d_err










    