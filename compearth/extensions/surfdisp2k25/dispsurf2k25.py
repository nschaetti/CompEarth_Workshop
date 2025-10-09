from __future__ import annotations

"""
Python implementation of the surfdisp96 subroutine from surfdisp96.f90.

This module provides a Python implementation of the surfdisp96 subroutine
from the surfdisp96 Fortran code, which is responsible for calculating
surface wave dispersion curves.
"""

from typing import Tuple, List, Union, Optional, Any
import numpy as np
import torch
import tqdm
import compearth.extensions.surfdisp2k25 as sd2k25


def dispsurf2k25_simulator(
        theta: torch.Tensor,
        p_min: float,
        p_max: float,
        kmax: int,
        iflsph: int = 0,
        iwave: int = 2,
        mode: int = 1,
        igr: int = 1,
        dtype: torch.dtype = torch.float32,
        progress: bool = False,
) -> torch.Tensor:
    """
    Compute dispersion curves using the dispsurf2k25 simulator.
    If kmax > 60, interpolation is performed to match the requested number of periods.

    Parameters
    ----------
    theta : torch.Tensor
        Model parameters [n_layers, vpvs, h_1...h_Nmax, vs_1...vs_Nmax], shape (B, D)
    p_min, p_max : float
        Minimum and maximum periods (s)
    kmax : int
        Desired number of output periods (interpolated if > 60)
    iflsph, iwave, mode, igr : int
        Physical simulation flags (see surfdisp96 documentation)
    dtype : torch.dtype
        Output tensor type (default: float32)
    progress : bool
        If True, display a tqdm progress bar.

    Returns
    -------
    torch.Tensor
        Simulated dispersion curves, shape (B, kmax)
    """

    bs = theta.shape[0]
    theta_np = theta.detach().cpu().numpy()

    # surfdisp2k25 only supports NP = 60 internally
    kmax_internal = 60
    t_internal = np.linspace(p_min, p_max, kmax_internal)
    output_disp_internal = np.zeros((bs, kmax_internal))

    # Progress bar
    iterator = tqdm.tqdm(range(bs), desc="Running dispsurf2k25", disable=not progress)

    for b_i in iterator:
        thkm = np.zeros(100)
        vpm = np.zeros(100)
        vsm = np.zeros(100)
        rhom = np.zeros(100)

        max_layer = (theta_np.shape[1] - 2) // 2
        n = int(theta_np[b_i, 0])
        vpvs = theta_np[b_i, 1]

        h = theta_np[b_i, 2:max_layer + 2]
        vs = theta_np[b_i, max_layer + 2:]

        vp = vs * vpvs
        rho = 0.32 + 0.77 * vp

        thkm[:n] = h[:n]
        vpm[:n] = vp[:n]
        vsm[:n] = vs[:n]
        rhom[:n] = rho[:n]

        disp_y, err = dispsurf2k25(
            thkm=thkm,
            vsm=vsm,
            vpm=vpm,
            rhom=rhom,
            nlayer=n,
            iflsph=iflsph,
            iwave=iwave,
            mode=mode,
            igr=igr,
            kmax=kmax_internal,
            t=t_internal,
        )

        if err != 0:
            raise RuntimeError(f"Simulation {b_i} failed with error: {err}")
        # end if

        output_disp_internal[b_i, :] = disp_y
    # end for

    # Interpolate to requested kmax (if different)
    if kmax != kmax_internal:
        t_target = np.linspace(p_min, p_max, kmax)
        output_disp = np.zeros((bs, kmax))
        for i in range(bs):
            output_disp[i, :] = np.interp(
                t_target,
                t_internal,
                output_disp_internal[i, :]
            )
        # end for
    else:
        output_disp = output_disp_internal
    # end if

    return torch.from_numpy(output_disp).to(dtype)
# end dispsurf2k25_simulator


def dispsurf2k25(
        thkm: np.ndarray,
        vpm: np.ndarray,
        vsm: np.ndarray,
        rhom: np.ndarray,
        nlayer: int,
        iflsph: int,
        iwave: int,
        mode: int,
        igr: int,
        kmax: int,
        t: np.ndarray
) -> Tuple[np.ndarray, int]:
    """
    Calculate surface wave dispersion curves.
    
    This is a pure Python implementation of the surfdisp96 Fortran subroutine.
    
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
    # Parameters
    LER = 0
    LIN = 5
    LOT = 6
    NL = 100
    NLAY = 100
    NL2 = NL + NL
    NP = 60

    # Check sizes
    assert thkm.ndim == 1 and thkm.shape[0] == NLAY, f"thkm must be 1dim (but is {thkm.ndim}), and shape {NLAY} (but is {thkm.shape})."
    assert vpm.ndim == 1 and vpm.shape[0] == NLAY, f"vpm must be 1dim (but is {vpm.ndim}), and shape {NLAY} (but is {vpm.shape})."
    assert vsm.ndim == 1 and vsm.shape[0] == NLAY, f"vsm must be 1dim (but is {vsm.ndim}), and shape {NLAY} (but is {vsm.shape})."
    assert rhom.ndim == 1 and rhom.shape[0] == NLAY, f"rhom must be 1dim (but is {rhom.ndim}), and shape {NLAY} (but is {rhom.shape})."
    assert t.ndim == 1 and t.shape[0] == NP, f"t must be 1dim (but is {t.ndim}), and shape {NP} (but is {t.shape})."

    # Arguments
    cg = np.zeros(NP, dtype=np.float64)

    # Local variables
    c = np.zeros(NP, dtype=np.float64)
    cb = np.zeros(NP, dtype=np.float64)
    d = np.zeros(NL, dtype=np.float64)
    a = np.zeros(NL, dtype=np.float64)
    b = np.zeros(NL, dtype=np.float64)
    rho = np.zeros(NL, dtype=np.float64)
    rtp = np.zeros(NL, dtype=np.float64)
    dtp = np.zeros(NL, dtype=np.float64)
    btp = np.zeros(NL, dtype=np.float64)
    iverb = np.zeros(3, dtype=np.int32)

    mmax = nlayer
    nsph = iflsph
    err = 0

    # Save current values
    b[:mmax] = vsm[:mmax]
    a[:mmax] = vpm[:mmax]
    d[:mmax] = thkm[:mmax]
    rho[:mmax] = rhom[:mmax]
    
    # Check if iwave is valid
    if iwave not in [1, 2]:
        raise ValueError(
            "iwave must be 1 (Love waves) or 2 (Rayleigh waves)"
        )
    # end if
    
    # Set up wave type
    if iwave == 1:
        idispl = kmax
        idispr = 0
    elif iwave == 2:
        idispl = 0
        idispr = kmax
    else:
        raise ValueError("iwave must be 1 or 2")
    # end if

    iverb[1] = 0
    iverb[2] = 0
    
    # Constants
    sone0 = 1.500
    ddc0 = 0.005
    h0 = 0.005

    # Check for water layer
    llw = 1
    if b[0] <= 0.0:
        llw = 2
    # end if
    twopi = 2.0 * np.pi
    one = 1.0e-2
    # Apply spherical earth transformation if needed
    if nsph == 1:
        d, a, b, rho, rtp, dtp, btp = sd2k25.sphere(
            ifunc=0,
            iflag=0,
            d=d,
            a=a,
            b=b,
            rho=rho,
            rtp=rtp,
            dtp=dtp,
            btp=btp,
            mmax=mmax,
            llw=llw,
            twopi=twopi
        )
    # end if
    
    JMN = 0
    betmx = -1.0e20
    betmn = 1.0e20
    # Find the extremal velocities to assist in starting search
    for i in range(nlayer):
        if 0.01 < b[i] < betmn:
            betmn = b[i]
            JMN = i
            jsol = 1
        elif b[i] <= 0.01 and a[i] < betmn:
            betmn = a[i]
            JMN = i
            jsol = 0
        # end if
        
        if b[i] > betmx:
            betmx = b[i]
        # end if
    # end for
    # Loop over wave types
    for ifunc in [1, 2]:
        if ifunc == 1 and idispl <= 0:
            continue
        # end if

        if ifunc == 2 and idispr <= 0:
            continue
        # end if
        
        # Apply spherical earth transformation for current wave type
        if nsph == 1:
            d, a, b, rho, rtp, dtp, btp = sd2k25.sphere(
                ifunc=ifunc,
                iflag=1,
                d=d,
                a=a,
                b=b,
                rho=rho,
                rtp=rtp,
                dtp=dtp,
                btp=btp,
                mmax=mmax,
                llw=llw,
                twopi=twopi
            )
        # end if
        ddc = ddc0
        sone = sone0
        h = h0
        
        if sone < 0.01:
            sone = 2.0
        # end if
        
        onea = sone
        
        # Get starting value for phase velocity
        if jsol == 0:
            # Water layer
            cc1 = betmn
        else:
            # Solid layer solves halfspace period equation
            cc1 = sd2k25.getsolh(
                a=float(a[JMN]),
                b=float(b[JMN])
            )
        # end if
        # Back off a bit to get a starting value at a lower phase velocity
        cc1 = 0.95 * cc1
        cc1 = 0.90 * cc1
        cc = cc1
        dc = ddc
        dc = abs(dc)
        c1 = cc
        cm = cc
        
        # Initialize arrays
        cb[:kmax] = 0.0
        c[:kmax] = 0.0
        
        ift = 999
        # Loop over modes
        # Warning: iq is from [1...mode]
        for iq in range(1, mode + 1):
            is_ = 0
            ie = kmax
            itst = ifunc
            
            # Loop over periods
            for k in range(is_, ie):
                if k >= ift:
                    break
                # end if

                # Get the period
                t1 = t[k]

                if igr > 0:
                    t1a = t1 / (1.0 + h)
                    t1b = t1 / (1.0 - h)
                    t1 = t1a
                else:
                    t1a = t1
                # end if igr > 0
                
                # Get initial phase velocity estimate to begin search
                if k == is_ and iq == 1:
                    c1 = cc
                    clow = cc
                    ifirst = 1
                elif k == is_ and iq > 1:
                    c1 = c[is_] + one * dc
                    clow = c1
                    ifirst = 1
                elif k > is_ and iq > 1:
                    ifirst = 0
                    clow = c[k] + one * dc
                    c1 = c[k-1]
                    if c1 < clow:
                        c1 = clow
                    # end if
                elif k > is_ and iq == 1:
                    ifirst = 0
                    c1 = c[k-1] - onea * dc
                    clow = cm
                else:
                    raise ValueError(f"Impossible to get initial phase velocity")
                # end if
                
                # Bracket root and refine it
                c1, iret = sd2k25.getsol(
                    t1=t1,
                    c1=c1,
                    clow=clow,
                    dc=dc,
                    cm=cm,
                    betmx=betmx,
                    ifunc=ifunc,
                    ifirst=ifirst,
                    d=d,
                    a=a,
                    b=b,
                    rho=rho,
                    rtp=rtp,
                    dtp=dtp,
                    btp=btp,
                    mmax=mmax,
                    llw=llw
                )
                
                if iret == -1:
                    break
                # end if
                
                c[k] = c1
                
                # For group velocities compute near above solution
                if igr > 0:
                    t1 = t1b
                    ifirst = 0
                    clow = cb[k] + one * dc
                    c1 = c1 - onea * dc
                    c1, iret = sd2k25.getsol(
                        t1=t1,
                        c1=c1,
                        clow=clow,
                        dc=dc,
                        cm=cm,
                        betmx=betmx,
                        ifunc=ifunc,
                        ifirst=ifirst,
                        d=d,
                        a=a,
                        b=b,
                        rho=rho,
                        rtp=rtp,
                        dtp=dtp,
                        btp=btp,
                        mmax=mmax,
                        llw=llw
                    )
                    
                    # Test if root not found at slightly larger period
                    if iret == -1:
                        c1 = c[k]
                    # end if
                    
                    cb[k] = c1
                else:
                    c1 = 0.0
                # end if igr
                
                cc0 = c[k]
                cc1 = c1
                
                if igr == 0:
                    # Output only phase velocity
                    cg[k] = cc0
                else:
                    # Calculate group velocity and output phase and group velocities
                    gvel = (1.0 / t1a - 1.0 / t1b) / (1.0 / (t1a*cc0) - 1.0 / (t1b*cc1))
                    cg[k] = gvel
                # end if igr
            # end for k is_..ie

            # If we broke out of the loop early
            if iverb[ifunc] == 0 and iq <= 1:
                # raise RuntimeError(f"Error, loop finished to early")
                pass
            # end if
            
            ift = k
            itst = 0
            
            # Set the remaining values to 0
            # for i in range(k, ie):
            #     t1a = t[i]
            #     cg[i] = 0.0
            # # end for k...ie

        # end for iq 1...mode
    # end for ifunc
    
    return cg, err
# end def dispsurf2k25
