

import numpy as np


def ppc(
        samples,
        z,
        disp_obs,
        n_keep=5,
        penalty_range,   # Interval sampling penality
        min_p,
        max_p,
        vpvs=1.75,
        random_state=None,
        return_all=False,
):
    """
    Run simulations for a sample (Posterior Predictive Check)
    """
    rng = np.random.default_rng(random_state)
    pers_obs = np.linspace(min_p, max_p, disp_obs.shape[0])

    results = []
    for i in range(samples.shape[0]):
        # Sample penalty from uniform distribution
        penalty = rng.uniform(*penalty_range)

        vs_flat, vsm, vpm, rhom, thkm = to_surfdisp_model(
            z,
            samples[i],
            penalty=penalty,
            vpvs=vpvs
        )

        n_layers = thkm.shape[0]
        pers_sim, dispvel_sim, err = forward_dispersion(
            vsm,
            vpm,
            rhom,
            thkm,
            n_layers,
            kmax=kmax,
            min_p=min_p,
            max_p=max_p
        )

        # === interpolation sur les périodes de l'obs ===
        dispvel_interp = np.interp(pers_obs, pers_sim, dispvel_sim)

        # === erreur RMS ===
        rms = np.sqrt(np.mean((dispvel_interp - disp_obs)**2))

        results.append({
            "idx": i,
            "penalty": penalty,
            "vs_flat": vs_flat,
            "vsm": vsm,
            "vpm": vpm,
            "rhom": rhom,
            "thkm": thkm,
            "dispvel": dispvel_interp,
            "pers": pers_obs,
            "rms": rms
        })
    # end for

    results_sorted = sorted(results, key=lambda r: r["rms"])

    if return_all:
        return results_sorted, results_sorted[:n_keep]
    # end if

    return results_sorted[:n_keep]
# end def ppc

