

import torch
import numpy as np

from compearth.extensions.surfdisp2k25 import dispsurf2k25_simulator

from .sampling import posterior_sample_to_theta


def ppc(
        posterior,
        n_samples,
        x_obs,
        z,
        penalty_min,
        penalty_max,   # Interval sampling penality
        min_p,
        max_p,
        layers_max,
        vpvs=1.75,
        best_k=100,
        rng=None,
        return_all=False,
        seed: int = 42
):
    """
    Run simulations for a sample (Posterior Predictive Check)
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    # end if

    iflsph = 0  # Flat Earth approximation (0 = flat, 1 = spherical)
    iwave = 2  # Wave type (2 = Rayleigh, 1 = Love)
    mode = 1  # Fundamental mode
    igr = 1  # Compute group velocity (1 = group, 0 = phase)

    # Periods
    pers_obs = np.linspace(min_p, max_p, x_obs.shape[0])

    # Generate samples
    samples = posterior.sample((n_samples,), x=x_obs)

    # Transform our posterior samples to theta vectors
    theta_models, _, _ = posterior_sample_to_theta(
        z=z,
        vs_batch=samples,
        vpvs=vpvs,
        penalty_min=penalty_min,
        penalty_max=penalty_max,
        max_layers=layers_max,
        rng=rng
    )

    # Run the simulator on the sampled models
    disp_curves = dispsurf2k25_simulator(
        theta=theta_models,
        p_min=min_p,
        p_max=max_p,
        kmax=x_obs.shape[0],
        iflsph=iflsph,
        iwave=iwave,
        mode=mode,
        igr=igr,
        progress=True
    )

    results = []
    for i in range(samples.shape[0]):
        # === erreur RMS ===
        rms = np.sqrt(torch.mean((disp_curves[0].cpu() - x_obs.cpu()) ** 2))
        results.append({
            "disp_curve": disp_curves[i],
            "rms": rms
        })
    # end for

    # Sort my RMS
    results_sorted = sorted(results, key=lambda r: r["rms"])
    best = results_sorted[:best_k]

    if return_all:
        return results_sorted, best
    # end if

    return best
# end def ppc

