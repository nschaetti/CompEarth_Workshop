# surfdisp2k25 Extension

This package provides a Python implementation of the surfdisp96 Fortran code, which is used for calculating dispersion curves for surface waves in layered media.

## Overview

The `surfdisp2k25` package includes Python implementations of several key functions from the original Fortran code:

- `dltar`: Computes the period equation for Love or Rayleigh waves
- `dltar1`: Computes the period equation for Love waves
- `dltar4`: Computes the period equation for Rayleigh waves
- `nevill`: Hybrid method for refining a root once it has been bracketed
- `half`: Interval halving method for refining a root
- `getsol`: Brackets a dispersion curve and then refines it

## Using the `getsol` Function

The `getsol` function is used to find phase velocities for surface waves at a given period. It first brackets the solution by finding values with opposite signs of the period equation, then refines the solution using the `nevill` function.

### Function Signature

```python
def getsol(t1, c1, clow, dc, cm, betmx, ifunc, ifirst, d, a, b, rho):
    """
    Bracket dispersion curve and then refine it.
    
    Parameters
    ----------
    t1 : float
        Period in seconds.
    c1 : float
        Initial phase velocity estimate in km/s.
    clow : float
        Lower bound for phase velocity in km/s.
    dc : float
        Phase velocity increment for search in km/s.
    cm : float
        Minimum phase velocity to consider in km/s.
    betmx : float
        Maximum phase velocity to consider in km/s.
    ifunc : int
        Wave type: 1 for Love waves, 2 for Rayleigh waves.
    ifirst : int
        First call flag: 1 for first call, 0 otherwise.
    d : array_like
        Layer thicknesses in km.
    a : array_like
        P-wave velocities in km/s.
    b : array_like
        S-wave velocities in km/s.
    rho : array_like
        Densities in g/cm^3.
    
    Returns
    -------
    tuple
        A tuple containing:
        - c1: float, the refined phase velocity in km/s.
        - iret: int, return code (1 for success, -1 for failure).
    """
```

### Example Usage

```python
import numpy as np
from migrate.extensions.surfdisp2k25 import getsol

# Define a three-layer model
d = np.array([2.0, 5.0, 10.0])  # Layer thicknesses in km
a = np.array([5.8, 6.8, 8.0])   # P-wave velocities in km/s
b = np.array([3.2, 3.9, 4.5])   # S-wave velocities in km/s
rho = np.array([2.7, 3.0, 3.3]) # Densities in g/cm^3

# Parameters for getsol
t1 = 10.0        # Period in seconds
c1 = 3.0         # Initial phase velocity estimate in km/s
clow = 2.0       # Lower bound for phase velocity
dc = 0.01        # Phase velocity increment
cm = 2.0         # Minimum phase velocity
betmx = 5.0      # Maximum phase velocity
ifunc = 2        # 2 for Rayleigh waves, 1 for Love waves
ifirst = 1       # 1 for first call, 0 otherwise

# Call getsol to find the phase velocity
c_refined, iret = getsol(t1, c1, clow, dc, cm, betmx, ifunc, ifirst, d, a, b, rho)

if iret == 1:
    print(f"Found phase velocity: {c_refined} km/s")
else:
    print("Failed to find a solution")
```

## Differences from the Original Fortran Code

The Python implementation differs from the original Fortran code in several ways:

1. **Return Values**: Instead of using output arguments, the Python functions return values directly. For example, `getsol` returns a tuple containing the refined phase velocity and a return code.

2. **Error Handling**: The Python implementation includes proper error handling with descriptive error messages.

3. **Array Handling**: The Python implementation uses NumPy arrays instead of Fortran arrays, which provides more flexibility and better performance.

4. **Static Variables**: The Fortran code uses static variables to save state between calls. The Python implementation simulates this using function attributes.

5. **Control Flow**: The Python implementation replaces goto statements with more structured control flow constructs like loops and conditional statements.

## Implementation Notes

- The `dltar`, `dltar1`, and `dltar4` functions are different methods for calculating the period equation for different wave types.
- The `nevill` function uses a combination of Neville's algorithm and interval halving to refine a root.
- The `half` function implements a simple interval halving method for root finding.
- The `getsol` function combines these methods to find phase velocities for dispersion curves.