"""
Transform spherical earth to flat earth.

This module contains the sphere function, which is a pure Python implementation
of the sphere_ Fortran subroutine.
"""
import math

import numpy as np
from typing import Tuple, List, Union, Optional

# Global variable to store dhalf between calls
_dhalf = 0.0

def sphere(
        ifunc: int,
        iflag: int,
        d: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        rho: np.ndarray,
        rtp: np.ndarray,
        dtp: np.ndarray,
        btp: np.ndarray,
        mmax: int,
        llw: int,
        twopi: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform spherical earth to flat earth

    Schwab, F. A., and L. Knopoff (1972). Fast surface wave and free
    mode computations, in Methods in Computational Physics, Volume 11,
    Seismology: Surface Waves and Earth Oscillations, B. A. Bolt (ed),
    Academic Press, New York

    Love Wave Equations 44, 45, 41 pp 112-113
    Rayleigh Wave Equations 102, 108, 109 pp 142, 144

    Revised 28 DEC 2007 to use mid-point, assume linear variation in
    slowness instead of using average velocity for the layer
    Use the Biswas (1972:PAGEOPH 96, 61-74, 1972) density mapping
    
    This is a pure Python implementation of the sphere_ Fortran subroutine.
    
    Parameters
    ----------
    ifunc : int
        Function type: 1 for Love waves, 2 for Rayleigh waves.
    iflag : int
        Flag: 0 for initialization, 1 for subsequent calls.
    d : array_like
        Layer thicknesses in km.
    a : array_like
        P-wave velocities in km/s.
    b : array_like
        S-wave velocities in km/s.
    rho : array_like
        Densities in g/cm^3.
    rtp : array_like, optional
        Original densities from forward transformation, used for backward transformation.
    dtp : array_like, optional
        Original thicknesses from forward transformation, used for backward transformation.
    btp : array_like, optional
        Transformation factors from forward transformation, used for backward transformation.
    
    Returns
    -------
    tuple
        A tuple containing:
        - d_new: array_like, transformed layer thicknesses
        - a_new: array_like, transformed P-wave velocities
        - b_new: array_like, transformed S-wave velocities
        - rho_new: array_like, transformed densities
        - rtp: array_like, original densities
        - dtp: array_like, original thicknesses
        - btp: array_like, transformation factors
    """
    global _dhalf
    NL = 100
    NP = 60

    # Check size
    assert d.ndim == 1 and d.shape[0] == NL, f"d! {d.shape[0]} != {NL}"
    assert a.ndim == 1 and a.shape[0] == NL, f"a! {a.shape[0]} != {NL}"
    assert b.ndim == 1 and b.shape[0] == NL, f"b! {b.shape[0]} != {NL}"
    assert rho.ndim == 1 and rho.shape[0] == NL, f"rho! {rho.shape[0]} != {NL}"
    assert rtp.ndim == 1 and rtp.shape[0] == NL, f"rtp! {rtp.shape[0]} != {NL}"
    assert dtp.ndim == 1 and dtp.shape[0] == NL, f"dtp! {dtp.shape[0]} != {NL}"
    assert btp.ndim == 1 and btp.shape[0] == NL, f"b! {btp.shape[0]} != {NL}"

    ar = 6370.0
    dr = 0.0
    r0 = ar
    d[mmax] = 1.0
    
    # Check array lengths
    if len(a) != mmax or len(b) != mmax or len(rho) != mmax:
        raise ValueError("d, a, b, and rho must have the same length")
    
    # Check if ifunc and iflag are valid
    if ifunc not in [1, 2]:
        raise ValueError("ifunc must be 1, or 2")
    # end if

    if iflag not in [0, 1]:
        raise ValueError("iflag must be 0 or 1")
    # end if
    
    # Forward transformation (spherical to flat)
    if iflag == 0:
        # Save original values
        for i in range(mmax):
            rtp[i] = rho[i]
            dtp[i] = d[i]
        # end for
        
        # Compute transformation factors
        for i in range(mmax):
            dr += d[i]
            r1 = ar - dr
            z0 = ar * math.log(ar / r0)
            z1 = ar * math.log(ar / r1)
            d[i] = z1 - z0
        # end for
        
        # Save the half-space depth
        _dhalf = d[mmax]
    # Backward transformation (flat to spherical)
    else:  # iflag == 1
        d[mmax] = _dhalf
        # Restore original values
        for i in range(mmax):
            if ifunc == 1:
                rho[i] = rtp[i] * btp[i]**(-5)
            elif ifunc == 2:
                rho[i] = rtp[i] * btp[i]**(-2.275)
            else:
                raise ValueError("ifunc must be 1 or 2")
            # end if
        # end for
    # end if

    d[mmax] = 0.0
    # Return the transformed arrays and the original values
    return d, a, b, rho, rtp, dtp, btp
# end def sphere
