import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, jvp, kv, kvp
from scipy.optimize import brentq

class CompactFiberSolver:
    def __init__(self, wavelength=1550e-9, l=1):
        self.wavelength = wavelength
        self.l = l
        self.k0 = 2 * np.pi / wavelength
        
    def sellmeier_silica(self, lam_um):
        """Calculate fused silica refractive index using Sellmeier equation"""
        lam2 = lam_um**2
        n_squared = (1 + 
                    0.6961663 * lam2 / (lam2 - 0.0684043**2) +
                    0.4079426 * lam2 / (lam2 - 0.1162414**2) +
                    0.8974794 * lam2 / (lam2 - 9.896161**2))
        return np.sqrt(n_squared)
    
    def algebraic_equation(self, beta, n1, n2, a):
        """Characteristic equation for fiber modes"""
        k1 = self.k0 * n1
        k2 = self.k0 * n2
        
        if not (k2 < beta < k1):
            return np.inf
            
        h = np.sqrt(k1**2 - beta**2)
        q = np.sqrt(beta**2 - k2**2)
        ha, qa = h * a, q * a
        
        try:
            Jl = jv(self.l, ha)
            Jlp = jvp(self.l, ha, 1)
            Kl = kv(self.l, qa)
            Klp = kvp(self.l, qa, 1)
            
            if abs(Jl) < 1e-12 or abs(Kl) < 1e-12:
                return np.inf
                
            term1 = Jlp / (ha * Jl) + Klp / (qa * Kl)
            term2 = n1**2 * Jlp / (ha * Jl) + n2**2 * Klp / (qa * Kl)
            term3 = self.l**2 * ((1/ha)**2 + (1/qa)**2)**2
            term4 = (beta / self.k0)**2
            
            return term1 * term2 - term3 * term4
        except:
            return np.inf
    
    def find_beta(self, n1, n2, a, max_iter=10, eps=1e-10):
        """Find fundamental mode propagation constant with iterative refinement"""
        k1, k2 = self.k0 * n1, self.k0 * n2
        beta_min = k2 * 1.0001
        beta_max = k1 * 0.9999
        
        # Initial coarse scan
        n_points = 1000
        beta_scan = np.linspace(beta_min, beta_max, n_points)
        eq_vals = [self.algebraic_equation(b, n1, n2, a) for b in beta_scan]
        
        # Find first zero crossing for initial bracket
        initial_bracket = None
        for i in range(len(eq_vals)-1):
            if eq_vals[i] * eq_vals[i+1] < 0:
                initial_bracket = (beta_scan[i], beta_scan[i+1])
                break
        
        if initial_bracket is None:
            return None
        
        # Iterative refinement around zero crossing
        beta_min_iter, beta_max_iter = initial_bracket
        beta_old = (beta_min_iter + beta_max_iter) / 2
        
        for iteration in range(max_iter):
            # Current range width
            range_width = beta_max_iter - beta_min_iter
            
            # Find refined root using Brent's method
            try:
                beta_new = brentq(lambda b: self.algebraic_equation(b, n1, n2, a),
                                beta_min_iter, beta_max_iter, xtol=eps/10)
            except:
                break
            
            # Check convergence
            if abs(beta_new - beta_old) < eps:
                return beta_new
            
            # Narrow the range for next iteration (10% of current range)
            new_range = 0.1 * range_width
            beta_min_iter = max(beta_new - new_range/2, k2 * 1.0001)
            beta_max_iter = min(beta_new + new_range/2, k1 * 0.9999)
            beta_old = beta_new
        
        return beta_old if 'beta_old' in locals() else None
    
    def beta_from_radius(self, n1, n2, a, max_iter=10, eps=1e-10):
        """半径 a (m) に対する基底モードの beta を返す（ラッパー）"""
        return self.find_beta(n1, n2, a, max_iter=max_iter, eps=eps)

    def diameter_from_beta(self, beta_target, n1, n2,
                           a_min_um=0.05, a_max_um=10.0,  # 半径の探索範囲 [μm]
                           n_scan=400, max_iter=20, eps=1e-10):
        """
        指定した beta_target から直径 d を返す（m）。
        - beta_target は k0*n2 と k0*n1 の間にある必要がある（導波条件）
        - 探索半径範囲は [a_min_um, a_max_um] μm でスキャンして符号反転区間を自動抽出
        """
        k1 = self.k0 * n1
        k2 = self.k0 * n2
        if not (k2 < beta_target < k1):
            raise ValueError(f"beta_target must be in (k0*n2, k0*n1) = ({k2:.6e}, {k1:.6e}).")

        # まず粗いスキャンで F(a)=beta(a)-beta_target の符号反転区間を見つける
        a_grid = np.linspace(a_min_um*1e-6, a_max_um*1e-6, n_scan)
        F_vals = []
        for a in a_grid:
            b = self.beta_from_radius(n1, n2, a, max_iter=max_iter, eps=eps)
            if b is None or np.isnan(b):
                F_vals.append(np.nan)
            else:
                F_vals.append(b - beta_target)
        F_vals = np.array(F_vals)

        # 有効な点だけでゼロ交差を探索
        idx = np.where(~np.isnan(F_vals))[0]
        if len(idx) < 2:
            raise RuntimeError("Insufficient valid beta samples in the scan range.")

        bracket = None
        for i0, i1 in zip(idx[:-1], idx[1:]):
            f0, f1 = F_vals[i0], F_vals[i1]
            if np.sign(f0) == 0:
                bracket = (a_grid[i0] * 0.999, a_grid[i0] * 1.001)
                break
            if f0 * f1 < 0:
                bracket = (a_grid[i0], a_grid[i1])
                break

        if bracket is None:
            # 単調に近いが符号反転が見つからないケース：端での最小絶対値を返すこともできるが、
            # ここでは明示的に失敗を知らせる
            raise RuntimeError("No sign change found for F(a) in the given radius range. "
                               "Try widening a_min_um/a_max_um.")

        a_lo, a_hi = bracket

        # Brent 法で a を高精度化
        from scipy.optimize import brentq
        def F(a):
            b = self.beta_from_radius(n1, n2, a, max_iter=max_iter, eps=eps)
            if b is None or np.isnan(b):
                # 例外を投げて brentq に失敗させないよう、巨大値で返す
                return 1e9
            return b - beta_target

        a_root = brentq(F, a_lo, a_hi, xtol=eps, rtol=1e-8, maxiter=100)
        diameter = 2.0 * a_root  # [m]
        return diameter

def main():
    # Parameters
    wavelength = 1389e-9  # 1389 nm
    solver = CompactFiberSolver(wavelength, l=1)
    
    # Calculate silica refractive index at 1550 nm
    lam_um = wavelength * 1e6
    n_silica = solver.sellmeier_silica(lam_um)
    n_air = 1.0
    
    print(f"Wavelength: {wavelength*1e9:.0f} nm")
    print(f"Silica refractive index: {n_silica:.6f}")
    print(f"Air refractive index: {n_air:.6f}")
    
    # Radius range: 0.3 to 1.0 μm
    radii_um = np.linspace(0.2, 0.8, 100)
    radii_m = radii_um * 1e-6
    
    # Calculate beta for each radius
    betas = []
    effective_indices = []
    
    for a in radii_m:
        beta = solver.find_beta(n_silica, n_air, a, max_iter=10, eps=1e-10)
        if beta is not None:
            betas.append(beta)
            effective_indices.append(beta / solver.k0)
        else:
            betas.append(np.nan)
            effective_indices.append(np.nan)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot beta vs radius
    ax1.plot(radii_um*2, np.array(betas) / solver.k0, 'b-o', markersize=3, linewidth=2)
    ax1.set_xlabel('Core Radius (μm)')
    ax1.set_ylabel('β/k₀ (Effective Index)')
    ax1.set_title(f'Propagation Constant vs Core Radius\n'
                 f'Silica core (n={n_silica:.4f}), Air cladding (n={n_air:.1f}), λ={wavelength*1e9:.0f} nm')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(radii_um[0]*2, radii_um[-1]*2)
    
    # Plot V-parameter
    V_params = solver.k0 * radii_m * np.sqrt(n_silica**2 - n_air**2)
    ax2.plot(radii_um*2, V_params, 'r-o', markersize=3, linewidth=2)
    ax2.axhline(y=2.405, color='k', linestyle='--', alpha=0.5, label='V=2.405 (cutoff)')
    ax2.set_xlabel('Core Radius (μm)')
    ax2.set_ylabel('V-parameter')
    ax2.set_title('V-parameter vs Core Radius')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(radii_um[0]*2, radii_um[-1]*2)
    
    plt.tight_layout()
    plt.show()
    
    # Print some results
    print(f"\nResults for selected radii:")
    print(f"{'Radius (μm)':<12} {'β/k₀':<12} {'V-param':<10}")
    print("-" * 35)
    for i in [0, 12, 24, 36, 49]:  # Selected indices
        if not np.isnan(betas[i]):
            print(f"{radii_um[i]:<12.2f} {betas[i]/solver.k0:<12.6f} {V_params[i]:<10.3f}")

if __name__ == "__main__":
    main()