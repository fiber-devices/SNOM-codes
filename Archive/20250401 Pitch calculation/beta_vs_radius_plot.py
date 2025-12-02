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
                        a_init_um=0.10,      # 初期半径 [µm]
                        grow=1.7,            # レンジ拡大率（指数探索）
                        a_min_um=0.02,       # 物理下限 [µm]
                        a_max_um=50.0,       # 物理上限（広めに）[µm]
                        max_expand=60,       # 最大拡張回数
                        max_iter=25, eps=1e-12):
        """
        β_target から直径 d を返す（m）。
        スキャンではなく、指数探索で自動的に符号反転区間 [a_lo, a_hi] を作る。
        挟めないときは端点での最接近解を返す。
        """
        k1, k2 = self.k0*n1, self.k0*n2
        if not (k2 < beta_target < k1):
            raise ValueError("beta_target must be strictly between k0*n2 and k0*n1.")

        def F_of_a(a):
            b = self.beta_from_radius(n1, n2, a, max_iter=max_iter, eps=eps)
            if b is None or np.isnan(b):
                return np.nan
            return b - beta_target

        # 初期点（下側）を用意：小さめ半径から上方向に探索
        a_lo = max(a_min_um*1e-6, a_init_um*1e-6)
        f_lo = F_of_a(a_lo)

        # もし NaN なら、少し大きい半径へずらして有効点を探す
        expand_cnt = 0
        while (np.isnan(f_lo) or f_lo >= 0) and expand_cnt < max_expand:
            # neff は a↑ で増える（通常）→ 目標より小さくするため更に小半径も試す
            trial = a_lo / grow
            if trial < a_min_um*1e-6:
                break
            a_lo = trial
            f_lo = F_of_a(a_lo)
            expand_cnt += 1

        # 上側端も用意：大きめ半径へ指数拡張
        a_hi = max(a_lo*grow, (a_lo*1e6 + 0.02)*1e-6)  # a_hi > a_lo を担保
        f_hi = F_of_a(a_hi)
        expand_cnt = 0
        while (np.isnan(f_hi) or f_hi <= 0) and expand_cnt < max_expand:
            trial = min(a_hi * grow, a_max_um*1e-6)
            if trial == a_hi:
                break
            a_hi = trial
            f_hi = F_of_a(a_hi)
            expand_cnt += 1

        # ブラケットが取れた？
        if (not np.isnan(f_lo)) and (not np.isnan(f_hi)) and (f_lo < 0) and (f_hi > 0):
            from scipy.optimize import brentq
            def G(a):
                val = F_of_a(a)
                if np.isnan(val):
                    # 近傍での数値不連続に備えて、非常に小さな変動を与えて再評価
                    da = max(1e-12, 1e-3*a)
                    val = F_of_a(a + da)
                    if np.isnan(val):
                        return 1e9  # brentq の失敗を誘発してリトライさせる
                return val

            # 念のためレンジを少し締める（端の NaN/inf を避ける）
            a_lo_eff = max(a_lo*(1+1e-6), a_min_um*1e-6)
            a_hi_eff = min(a_hi*(1-1e-6), a_max_um*1e-6)

            a_root = brentq(G, a_lo_eff, a_hi_eff, xtol=eps, rtol=1e-9, maxiter=200)
            return 2.0 * a_root  # 直径 [m]

        # 挟めなかった：最接近解でフォールバック
        # 下側から一定本数サンプルし、|F| が最小の a を返す
        samples = []
        a = max(a_min_um*1e-6, 0.5 * a_lo)
        while a <= min(a_max_um*1e-6, max(a_hi, a_lo)*grow):
            f = F_of_a(a)
            if not np.isnan(f):
                samples.append((abs(f), a))
            a *= 1.15
            if len(samples) > 1200:
                break

        if len(samples) == 0:
            raise RuntimeError("Could not obtain valid samples; raise a_min_um a bit and/or reduce tiny-a region.")
        # 最接近点
        a_best = min(samples, key=lambda t: t[0])[1]
        return 2.0 * a_best


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