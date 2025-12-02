import numpy as np
from scipy.special import jv, jvp, kv, kvp
from scipy.optimize import root_scalar

class DiameterCalculation:
    def __init__(self, wavelength_nm, standing_wave_pitch_nm):
        self.wavelength_nm = wavelength_nm
        self.standing_wave_pitch_nm = standing_wave_pitch_nm
        self.n_air = 1.0
        self.n_glass = self.glass_n(wavelength_nm)

    def glass_n(self, wavelength_nm):
        """Sellmeier equation for fused silica."""
        wavelength_um = wavelength_nm * 1e-3
        B1, B2, B3 = 0.6961663, 0.4079426, 0.8974794
        C1, C2, C3 = 0.0684043**2, 0.1162414**2, 9.896161**2
        lam2 = wavelength_um**2
        n2 = 1 + B1 * lam2 / (lam2 - C1) + B2 * lam2 / (lam2 - C2) + B3 * lam2 / (lam2 - C3)
        return np.sqrt(n2)

    def eigenvalue_residual(self, D):
        """Residual of the eigenvalue equation (Eq. 2). D in meters."""
        λ0 = self.wavelength_nm * 1e-9
        Γ = self.standing_wave_pitch_nm * 1e-9
        k0 = 2 * np.pi / λ0
        beta = np.pi / Γ
        n_air = self.n_air
        n_fiber = self.n_glass
        m = 1

        # Check physical range of beta
        if not (k0 * n_air < beta < k0 * n_fiber):
            return np.inf

        u = (D / 2) * np.sqrt((k0 * n_fiber)**2 - beta**2)
        w = (D / 2) * np.sqrt(beta**2 - (k0 * n_air)**2)

        if u <= 0 or w <= 0:
            return np.inf

        try:
            lhs = m**2 * (1/u**2 + 1/w**2) * (1/u**2 + (n_air**2 / n_fiber**2) * 1/w**2)
            rhs = (
                (1/u * jvp(m, u)/jv(m, u) + 1/w * kvp(m, w)/kv(m, w)) *
                (1/u * jvp(m, u)/jv(m, u) + (n_air**2 / n_fiber**2) * 1/w * kvp(m, w)/kv(m, w))
            )
            return lhs - rhs
        except Exception:
            return np.inf

    def find_bracket(self, D_start_um=0.01, D_end_um=2.0, step_um=0.001):
        """Scan for valid bracket where eigenvalue_residual crosses zero."""
        x = D_start_um
        f_prev = self.eigenvalue_residual(x * 1e-6)
        for x_new in np.arange(D_start_um + step_um, D_end_um, step_um):
            f_new = self.eigenvalue_residual(x_new * 1e-6)
            if np.sign(f_prev) != np.sign(f_new):
                return (x * 1e-6, x_new * 1e-6)
            f_prev = f_new
            x = x_new
        # raise ValueError("No root bracket found in diameter range.")

    def solve_diameter(self):
        """Main solve method with automatic bracketing."""
        bracket = self.find_bracket()
        result = root_scalar(
            self.eigenvalue_residual,
            bracket=bracket,
            method='brentq',
            xtol=1e-10
        )
        if result.converged:
            return result.root * 1e6  # return in µm
        else:
            return np.nan
            # raise RuntimeError("Root finding did not converge.")

    def safe_solve_diameter(self):
        """Safe version that returns NaN if solving fails."""
        try:
            return self.solve_diameter()
        except Exception as e:
            return np.nan

