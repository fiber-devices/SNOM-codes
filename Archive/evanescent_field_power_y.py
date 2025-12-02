"""
Example script to estimate the optical power of the evanescent field 
in the y-direction (vertical component) of an optical fiber.

For HE11 mode, Ey = 0, so the power in the y-direction is zero.
This script demonstrates how to calculate the evanescent field components
and the power distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
from eeequation import (
    calculate_evanescent_field_components,
    calculate_evanescent_power_y_direction,
    calculate_beta,
    refractive_index_fused_silica
)

def main():
    # Fiber parameters
    diameter_um = 0.5  # Fiber diameter in micrometers
    wavelength_um = 1.550  # Wavelength in micrometers (1550 nm)
    n_fiber = refractive_index_fused_silica(wavelength_um)
    n_air = 1.0
    
    print(f"Fiber diameter: {diameter_um} µm")
    print(f"Wavelength: {wavelength_um} µm ({wavelength_um*1000} nm)")
    print(f"Fiber refractive index: {n_fiber:.6f}")
    print(f"Air refractive index: {n_air:.1f}")
    
    # Calculate propagation constant
    beta = calculate_beta(diameter_um, n_fiber, wavelength_um)
    if np.isnan(beta):
        print("Error: Could not calculate propagation constant")
        return
    
    print(f"Propagation constant β: {beta:.6f} rad/µm")
    print(f"Effective index n_eff = β/k0: {beta * wavelength_um / (2 * np.pi):.6f}")
    print()
    
    # Fiber radius
    a = (diameter_um / 2) * 1e-6  # Convert to meters
    
    # Radial distances for calculation (in meters)
    # Start from just outside the fiber core
    r_min = a * 1.01  # Slightly outside the core
    r_max = a * 3.0   # Up to 3 times the radius
    r = np.linspace(r_min, r_max, 1000)
    
    # Calculate evanescent field components
    Ex, Ey, Ez = calculate_evanescent_field_components(
        diameter_um, n_fiber, wavelength_um, r, Alin=1.0
    )
    
    # Calculate power in y-direction
    Py = calculate_evanescent_power_y_direction(
        diameter_um, n_fiber, wavelength_um, r, Alin=1.0
    )
    
    # Calculate total power density (for comparison)
    epsilon0 = 8.854e-12  # F/m
    c = 3e8  # m/s
    P_total = 0.5 * epsilon0 * c * (np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)
    Px = 0.5 * epsilon0 * c * np.abs(Ex)**2
    Pz = 0.5 * epsilon0 * c * np.abs(Ez)**2
    
    # Convert radial distance to micrometers for plotting
    r_um = r * 1e6
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Electric field components
    ax = axes[0, 0]
    ax.semilogy(r_um, np.abs(Ex), 'r-', label='|Ex|', linewidth=2)
    ax.semilogy(r_um, np.abs(Ey), 'g-', label='|Ey|', linewidth=2)
    ax.semilogy(r_um, np.abs(Ez), 'b-', label='|Ez|', linewidth=2)
    ax.axvline(x=diameter_um/2, color='k', linestyle='--', alpha=0.5, label='Fiber core edge')
    ax.set_xlabel('Radial distance from center (µm)')
    ax.set_ylabel('Electric field magnitude (V/m)')
    ax.set_title('Evanescent Field Components vs Radial Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Power density components
    ax = axes[0, 1]
    ax.semilogy(r_um, Px, 'r-', label='Power in x-direction', linewidth=2)
    ax.semilogy(r_um, Py, 'g-', label='Power in y-direction', linewidth=2)
    ax.semilogy(r_um, Pz, 'b-', label='Power in z-direction', linewidth=2)
    ax.semilogy(r_um, P_total, 'k--', label='Total power', linewidth=2, alpha=0.7)
    ax.axvline(x=diameter_um/2, color='k', linestyle='--', alpha=0.5, label='Fiber core edge')
    ax.set_xlabel('Radial distance from center (µm)')
    ax.set_ylabel('Power density (W/m²)')
    ax.set_title('Power Density Components vs Radial Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Power in y-direction (should be zero for HE11)
    ax = axes[1, 0]
    ax.plot(r_um, Py, 'g-', linewidth=2)
    ax.axvline(x=diameter_um/2, color='k', linestyle='--', alpha=0.5, label='Fiber core edge')
    ax.set_xlabel('Radial distance from center (µm)')
    ax.set_ylabel('Power density in y-direction (W/m²)')
    ax.set_title('Power in Y-Direction (Vertical Component)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 4: Field decay (showing exponential decay)
    ax = axes[1, 1]
    ax.semilogy(r_um, np.abs(Ex), 'r-', label='|Ex|', linewidth=2)
    ax.semilogy(r_um, np.abs(Ez), 'b-', label='|Ez|', linewidth=2)
    ax.axvline(x=diameter_um/2, color='k', linestyle='--', alpha=0.5, label='Fiber core edge')
    ax.set_xlabel('Radial distance from center (µm)')
    ax.set_ylabel('Electric field magnitude (V/m)')
    ax.set_title('Evanescent Field Decay (Ex and Ez)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evanescent_field_power_y.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'evanescent_field_power_y.png'")
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"At radial distance r = {r[0]*1e6:.3f} µm (just outside core):")
    print(f"  |Ex| = {np.abs(Ex[0]):.6e} V/m")
    print(f"  |Ey| = {np.abs(Ey[0]):.6e} V/m")
    print(f"  |Ez| = {np.abs(Ez[0]):.6e} V/m")
    print(f"  Power in y-direction: {Py[0]:.6e} W/m²")
    print()
    print(f"At radial distance r = {r[-1]*1e6:.3f} µm:")
    print(f"  |Ex| = {np.abs(Ex[-1]):.6e} V/m")
    print(f"  |Ey| = {np.abs(Ey[-1]):.6e} V/m")
    print(f"  |Ez| = {np.abs(Ez[-1]):.6e} V/m")
    print(f"  Power in y-direction: {Py[-1]:.6e} W/m²")
    print()
    print("NOTE: For HE11 mode, Ey = 0, so the power in the y-direction is zero.")
    print("      The evanescent field has components only in the x and z directions.")

if __name__ == "__main__":
    main()


