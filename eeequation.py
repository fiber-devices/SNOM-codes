import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, kv, jvp, kvp
from scipy.optimize import root_scalar

# Function to calculate beta by solving the eigenvalue equation
def calculate_beta(diameter, n_fiber, wavelength):
    n_air = 1
    k0 = 2 * np.pi / wavelength  # Wavenumber in vacuum
    
    def eigenvalue_eq(beta):
        if beta > k0 * n_fiber or beta < k0 * n_air:
            return np.inf
        
        u = (diameter / 2) * np.sqrt(k0**2 * n_fiber**2 - beta**2)
        w = (diameter / 2) * np.sqrt(beta**2 - k0**2 * n_air**2)
        
        if u <= 0 or w <= 0:
            return np.inf
        
        lhs = ((1/u**2) + (1/w**2)) * ((1/u**2)+((n_air**2 / n_fiber**2) * (1/w**2)))
        rhs = (((1 / u) * (jvp(1, u) / jv(1, u))) + ((1 / w) * (kvp(1, w) / kv(1, w)))) * \
              (((1 / u) * (jvp(1, u) / jv(1, u))) + ((n_air**2 / n_fiber**2) * (1 / w) * (kvp(1, w) / kv(1, w))))
        return lhs - rhs
    
    beta_min = k0 * n_air + 1e-5
    beta_max = k0 * n_fiber - 1e-5
    
    if eigenvalue_eq(beta_min) * eigenvalue_eq(beta_max) > 0:
        return np.nan
    
    try:
        sol = root_scalar(eigenvalue_eq, bracket=[beta_min, beta_max], method='brentq', xtol=1e-6)
        return sol.root if sol.converged else np.nan
    except ValueError:
        return np.nan

#Sellmeir-type dispertions
def refractive_index_fused_silica(lamda):
    return np.sqrt(1 + ((0.6961663 * lamda**2)/((lamda**2)-(0.0684043)**2)) \
                   + ((0.4079426 * lamda**2) / ((lamda**2) - (0.1162414)**2))\
                      + ((0.8974794 * lamda**2)/(lamda**2 - (9.896161)**2)))

def calculate_evanescent_field_components(diameter, n_fiber, wavelength, r, Alin=1.0):
    """
    Calculate the evanescent field components (Ex, Ey, Ez) at a radial distance r from the fiber center.
    
    Parameters:
    -----------
    diameter : float
        Fiber diameter in micrometers
    n_fiber : float
        Refractive index of the fiber core
    wavelength : float
        Wavelength in micrometers
    r : float or array
        Radial distance from fiber center in meters (r > a, where a is the fiber radius)
    Alin : float
        Normalization constant (default: 1.0)
    
    Returns:
    --------
    Ex, Ey, Ez : float or array
        Electric field components in the evanescent region
        For HE11 mode: Ey = 0
    """
    n_air = 1.0
    a = (diameter / 2) * 1e-6  # Convert to meters
    wl = wavelength * 1e-6  # Convert to meters
    k0 = 2 * np.pi / wl
    
    # Calculate propagation constant
    beta = calculate_beta(diameter, n_fiber, wavelength)
    if np.isnan(beta):
        return np.nan, np.nan, np.nan
    
    # Convert beta to rad/m if needed (calculate_beta returns in rad/um)
    beta = beta * 1e6  # Convert to rad/m
    
    # Calculate h11 and q11
    h11 = np.sqrt(k0**2 * n_fiber**2 - beta**2)
    q11 = np.sqrt(beta**2 - k0**2 * n_air**2)
    
    # Calculate s11 parameter
    s11 = (1/(h11*a)**2 + 1/(q11*a)**2) * (jvp(1, h11*a)/(h11*a*jv(1, h11*a)) + kvp(1, q11*a)/(q11*a*kv(1, q11*a)))
    
    # Ensure r is an array for vectorized operations
    r = np.asarray(r)
    
    # Calculate field components in the evanescent region (r > a)
    Ex = Alin * beta * jv(1, h11 * a) / (2 * q11 * kv(1, q11 * a)) * ((1 - s11) * kv(0, q11 * r) + (1 + s11) * kv(2, q11 * r))
    Ey = 0  # For HE11 mode, Ey = 0
    Ez = Alin * jv(1, h11 * a) / kv(1, q11 * a) * kv(1, q11 * r)
    
    return Ex, Ey, Ez

def calculate_evanescent_power_y_direction(diameter, n_fiber, wavelength, r, Alin=1.0, epsilon0=8.854e-12, c=3e8):
    """
    Calculate the optical power density in the y-direction (vertical component) of the evanescent field.
    
    The power density is proportional to |Ey|^2. For HE11 mode, Ey = 0, so the power in the y-direction is zero.
    
    Parameters:
    -----------
    diameter : float
        Fiber diameter in micrometers
    n_fiber : float
        Refractive index of the fiber core
    wavelength : float
        Wavelength in micrometers
    r : float or array
        Radial distance from fiber center in meters (r > a, where a is the fiber radius)
    Alin : float
        Normalization constant (default: 1.0)
    epsilon0 : float
        Permittivity of free space (default: 8.854e-12 F/m)
    c : float
        Speed of light in vacuum (default: 3e8 m/s)
    
    Returns:
    --------
    Py : float or array
        Power density in the y-direction [W/m^2]
        For HE11 mode, this will be zero since Ey = 0
    """
    Ex, Ey, Ez = calculate_evanescent_field_components(diameter, n_fiber, wavelength, r, Alin)
    
    # Power density in y-direction is proportional to |Ey|^2
    # P_y = (1/2) * epsilon0 * c * |Ey|^2
    Py = 0.5 * epsilon0 * c * np.abs(Ey)**2
    
    return Py

# Function to calculate diameter by solving eigen equation
def calculate_diameter(pitch, n_fiber=1.44, wavelength=1.38825):
    pitch = pitch / 1000  # Convert pitch from mm to micrometers
    n_air = 1
    k0 = 2 * np.pi / wavelength  # Wavenumber in vacuum
    beta = np.pi / pitch

    def eigenvalue_eq(diameter):
        if beta > k0 * n_fiber or beta < k0 * n_air:
            return np.inf
        
        u = (diameter / 2) * np.sqrt(k0**2 * n_fiber**2 - beta**2)
        w = (diameter / 2) * np.sqrt(beta**2 - k0**2 * n_air**2)
        
        if u <= 0 or w <= 0:
            return np.inf
        
        lhs = ((1/u**2) + (1/w**2)) * ((1/u**2)+((n_air**2 / n_fiber**2) * (1/w**2)))
        rhs = (((1 / u) * (jvp(1, u) / jv(1, u))) + ((1 / w) * (kvp(1, w) / kv(1, w)))) * \
              (((1 / u) * (jvp(1, u) / jv(1, u))) + ((n_air**2 / n_fiber**2) * (1 / w) * (kvp(1, w) / kv(1, w))))
        return lhs - rhs
    
    diameter_min = 0.005
    diameter_max = 1.30
    
    if eigenvalue_eq(diameter_min) * eigenvalue_eq(diameter_max) > 0:
        return np.nan
    
    try:
        sol = root_scalar(eigenvalue_eq, bracket=[diameter_min, diameter_max], method='brentq', xtol=1e-6)
        return sol.root if sol.converged else np.nan
    except ValueError:
        return np.nan
    

def calculate_pitch(diameter, n_fiber, wavelength):
    n_air = 1
    k0 = 2 * np.pi / wavelength  # Wavenumber in vacuum
    
    def eigenvalue_eq(beta):
        if beta > k0 * n_fiber or beta < k0 * n_air:
            return np.inf
        
        u = (diameter / 2) * np.sqrt(k0**2 * n_fiber**2 - beta**2)
        w = (diameter / 2) * np.sqrt(beta**2 - k0**2 * n_air**2)
        
        if u <= 0 or w <= 0:
            return np.inf
        
        lhs = ((1/u**2) + (1/w**2)) * ((1/u**2)+((n_air**2 / n_fiber**2) * (1/w**2)))
        rhs = (((1 / u) * (jvp(1, u) / jv(1, u))) + ((1 / w) * (kvp(1, w) / kv(1, w)))) * \
              (((1 / u) * (jvp(1, u) / jv(1, u))) + ((n_air**2 / n_fiber**2) * (1 / w) * (kvp(1, w) / kv(1, w))))
        return lhs - rhs
    
    beta_min = k0 * n_air + 1e-5
    beta_max = k0 * n_fiber - 1e-5
    
    if eigenvalue_eq(beta_min) * eigenvalue_eq(beta_max) > 0:
        return np.nan
    
    try:
        sol = root_scalar(eigenvalue_eq, bracket=[beta_min, beta_max], method='brentq', xtol=1e-6)
        return np.pi / sol.root * 1000 if sol.converged else np.nan
    except ValueError:
        return np.nan