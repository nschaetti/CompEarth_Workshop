"""
Evaluate variables for the compound matrix.

This module contains the var function, which is a pure Python implementation
of the var_ Fortran subroutine.
"""

import numpy as np

def var(
        p: float,
        q: float,
        ra: float,
        rb: float,
        wvno: float,
        xka: float,
        xkb: float,
        dpth: float
):
    """
    Evaluate variables for the compound matrix.
    
    This is a pure Python implementation of the var_ Fortran subroutine.
    
    Parameters
    ----------
    p : float
        P-wave vertical slowness parameter (ra * depth).
    q : float
        S-wave vertical slowness parameter (rb * depth).
    ra : float
        P-wave vertical slowness.
    rb : float
        S-wave vertical slowness.
    wvno : float
        Horizontal wavenumber in rad/km.
    xka : float
        P-wave wavenumber (omega/alpha).
    xkb : float
        S-wave wavenumber (omega/beta).
    dpth : float
        Layer thickness in km.
    
    Returns
    -------
    tuple
        A tuple containing the following variables:
        (w, cosp, exa, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz)
    """
    # Initialize variables
    exa = 0.0
    a0 = 1.0
    
    # Examine P-wave eigenfunctions
    # checking whether c > vp, c = vp or c < vp
    pex = 0.0
    sex = 0.0
    
    if wvno < xka:
        sinp = np.sin(p)
        w = sinp / ra
        x = -ra * sinp
        cosp = np.cos(p)
    elif wvno == xka:
        cosp = 1.0
        w = dpth
        x = 0.0
    elif wvno > xka:
        pex = p
        fac = 0.0
        if p < 16:
            fac = np.exp(-2.0 * p)
        # end if
        cosp = (1.0 + fac) * 0.5
        sinp = (1.0 - fac) * 0.5
        w = sinp / ra
        x = ra * sinp
    # end if
    
    # Examine S-wave eigenfunctions
    # checking whether c > vs, c = vs, c < vs
    if wvno < xkb:
        sinq = np.sin(q)
        y = sinq / rb
        z = -rb * sinq
        cosq = np.cos(q)
    elif wvno == xkb:
        cosq = 1.0
        y = dpth
        z = 0.0
    elif wvno > xkb:
        sex = q
        fac = 0.0
        if q < 16:
            fac = np.exp(-2.0 * q)
        # end if
        cosq = (1.0 + fac) * 0.5
        sinq = (1.0 - fac) * 0.5
        y = sinq / rb
        z = rb * sinq
    # end if
    
    # Form eigenfunction products for use with compound matrices
    exa = pex + sex
    a0 = 0.0

    if exa < 60.0:
        a0 = np.exp(-exa)
    # end if

    cpcq = cosp * cosq
    cpy = cosp * y
    cpz = cosp * z
    cqw = cosq * w
    cqx = cosq * x
    xy = x * y
    xz = x * z
    wy = w * y
    wz = w * z
    qmp = sex - pex
    fac = 0.0

    if qmp > -40.0:
        fac = np.exp(qmp)
    # end if

    cosq = cosq * fac
    y = fac * y
    z = fac * z
    
    # Return all computed values as a tuple
    return w, cosp, exa, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz
# end def var
