import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_velocity_and_dispersion(
        thicknesses,
        vs_layers,
        disp_curve,
        p_min,
        p_max,
        kmax
):
    """
    Plot a layered velocity model (Vs vs depth) and its dispersion curve.

    Parameters
    ----------
    thicknesses : list[float]
        List of layer thicknesses (km). The last one can be 0 for half-space.
    vs_layers : list[float]
        List of Vs velocities (km/s), one per layer.
    disp_curve : torch.Tensor or np.ndarray
        Dispersion curve (group velocity as function of period).
    p_min : float
        Minimum period (s).
    p_max : float
        Maximum period (s).
    kmax : int
        Number of period samples.
    """
    # Convert dispersion curve to numpy
    if isinstance(disp_curve, torch.Tensor):
        disp_curve = disp_curve.detach().cpu().numpy().flatten()

    # Compute depth interfaces
    n_layers = len(vs_layers)
    finite_thicknesses = [t for t in thicknesses if t > 0]
    depth_interfaces = np.concatenate(([0], np.cumsum(finite_thicknesses)))
    max_depth = depth_interfaces[-1] + 1.0  # extend visually for half-space

    # Build step-like profile
    z_plot = [0]
    vs_plot = [vs_layers[0]]

    for i in range(n_layers - 1):
        z_plot.extend([depth_interfaces[i + 1], depth_interfaces[i + 1]])
        vs_plot.extend([vs_layers[i], vs_layers[i + 1]])

    # Extend half-space
    z_plot.append(max_depth)
    vs_plot.append(vs_layers[-1])

    z_plot = np.array(z_plot)
    vs_plot = np.array(vs_plot)

    # Prepare periods
    periods = np.linspace(p_min, p_max, kmax)

    # ---- Plot ----
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Velocity model
    ax[0].plot(z_plot, vs_plot, color='royalblue', linewidth=2)
    ax[0].fill_between(z_plot, 0, vs_plot, color='royalblue', alpha=0.2)
    ax[0].set_xlim(0, max_depth)
    ax[0].set_ylim(min(vs_layers) * 0.9, max(vs_layers) * 1.1)
    ax[0].set_xlabel("Depth [km]")
    ax[0].set_ylabel("Vs [km/s]")
    ax[0].set_title("Velocity model (layered)")
    ax[0].grid(True, linestyle='--', alpha=0.5)

    # Dispersion curve
    ax[1].plot(periods, disp_curve, color='tomato', linewidth=2)
    ax[1].set_xlabel("Period [s]")
    ax[1].set_ylabel("Group velocity [km/s]")
    ax[1].set_title("Dispersion curve")
    ax[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
# end def plot_velocity_and_dispersion
