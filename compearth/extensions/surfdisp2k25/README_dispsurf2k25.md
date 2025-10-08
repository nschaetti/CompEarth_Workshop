# dispsurf2k25 Function

## Overview

The `dispsurf2k25` function is a Python implementation of the Fortran subroutine `surfdisp96` from the `surfdisp96.f90` file. It calculates surface wave dispersion curves for either Love waves or Rayleigh waves.

## Implementation Details

The implementation follows these key principles:

1. **Pure Python Implementation**: The function is implemented in pure Python using NumPy for array operations.
2. **Return Values Instead of Output Arguments**: Unlike the Fortran subroutine which uses output arguments, the Python implementation returns values as a tuple.
3. **Python Implementations of Dependencies**: The function uses the Python implementations of `sphere`, `getsol`, and `getsolh` from the `surfdisp2k25` module.
4. **Input Validation**: The function includes validation of input parameters to ensure they are valid.

## Function Signature

```python
def dispsurf2k25(thkm, vpm, vsm, rhom, iflsph, iwave, mode, igr, kmax, t):
    """
    Calculate surface wave dispersion curves.
    
    Parameters
    ----------
    thkm : array_like
        Layer thicknesses in km.
    vpm : array_like
        P-wave velocities in km/s.
    vsm : array_like
        S-wave velocities in km/s.
    rhom : array_like
        Densities in g/cm^3.
    iflsph : int
        Flag for spherical earth model: 0 for flat earth, 1 for spherical earth.
    iwave : int
        Wave type: 1 for Love waves, 2 for Rayleigh waves.
    mode : int
        Mode number to calculate.
    igr : int
        Flag for group velocity calculation: 0 for phase velocity only, 1 for phase and group velocity.
    kmax : int
        Number of periods to calculate.
    t : array_like
        Periods in seconds.
    
    Returns
    -------
    tuple
        A tuple containing:
        - cg: array_like, calculated phase or group velocities in km/s.
        - err: int, error code (0 for success, 1 for error).
    """
```

## Usage Examples

### Basic Usage with Love Waves

```python
import numpy as np
from compearth.extensions.surfdisp2k25 import dispsurf2k25

# Create a simple model
# 3 layers: 2 layers + half-space
thkm = np.array([10.0, 20.0, 0.0])  # Layer thicknesses in km
vpm = np.array([5.0, 6.0, 7.0])  # P-wave velocities in km/s
vsm = np.array([3.0, 3.5, 4.0])  # S-wave velocities in km/s
rhom = np.array([2.7, 2.9, 3.1])  # Densities in g/cm^3

# Parameters for dispsurf2k25
iflsph = 0  # Flat earth model
iwave = 1  # Love waves
mode = 1  # Fundamental mode
igr = 0  # Phase velocity only
kmax = 5  # Number of periods
t = np.array([5.0, 10.0, 15.0, 20.0, 25.0])  # Periods in seconds

# Call the function
cg, err = dispsurf2k25(thkm, vpm, vsm, rhom, iflsph, iwave, mode, igr, kmax, t)
print(f"Phase velocities: {cg}")
print(f"Error code: {err}")
```

### Usage with Rayleigh Waves

```python
# Same model as above, but with Rayleigh waves
iwave = 2  # Rayleigh waves

# Call the function
cg, err = dispsurf2k25(thkm, vpm, vsm, rhom, iflsph, iwave, mode, igr, kmax, t)
print(f"Phase velocities: {cg}")
print(f"Error code: {err}")
```

### Usage with Group Velocity Calculation

```python
# Same model as above, but with group velocity calculation
iwave = 1  # Love waves
igr = 1    # Group velocity calculation

# Call the function
cg, err = dispsurf2k25(thkm, vpm, vsm, rhom, iflsph, iwave, mode, igr, kmax, t)
print(f"Group velocities: {cg}")
print(f"Error code: {err}")
```

### Usage with Spherical Earth Model

```python
# Same model as above, but with spherical earth model
iflsph = 1  # Spherical earth model
iwave = 1   # Love waves
igr = 0     # Phase velocity only

# Call the function
cg, err = dispsurf2k25(thkm, vpm, vsm, rhom, iflsph, iwave, mode, igr, kmax, t)
print(f"Phase velocities: {cg}")
print(f"Error code: {err}")
```

## Notes

- The function returns an error code of 1 if it encounters an error during the calculation, such as not finding a root for a particular period.
- For periods where a root is not found, the corresponding velocity in the output array is set to 0.0.
- The function handles both Love waves (iwave=1) and Rayleigh waves (iwave=2).
- The function can calculate either phase velocities only (igr=0) or both phase and group velocities (igr=1).
- The function can handle both flat earth models (iflsph=0) and spherical earth models (iflsph=1).