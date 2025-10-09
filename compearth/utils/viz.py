

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Union, List, Sequence, Optional


def plot_velocity_and_dispersion(
        theta: Union[torch.Tensor, np.ndarray],
        disp_curve: Union[torch.Tensor, np.ndarray],
        p_min: float,
        p_max: float,
        kmax: int,
        z_vnoi: Optional[Union[torch.Tensor, np.ndarray]] = None,
        fig_size: float = 2.0,
        dpi: int = 300
) -> None:
    """
    Plot one or several layered velocity models (Vs vs depth)
    and their corresponding dispersion curves, directly from θ.

    Parameters
    ----------
    theta : torch.Tensor or np.ndarray
        Model parameters of shape (B, 2 + 2*Nmax):
        [n_layers, vpvs, h_1...h_Nmax, vs_1...vs_Nmax]
    disp_curve : torch.Tensor or np.ndarray
        Dispersion curves of shape (B, kmax)
    p_min, p_max : float
        Minimum and maximum periods (s)
    kmax : int
        Number of period samples
    z_vnoi : torch.Tensor or np.ndarray or None, optional
        Voronoi midpoints (B, Nmax). If provided, non-zero values are plotted
        except the last (half-space) one.
    fig_size : float
        Scaling factor for figure size
    dpi : int
        Plot resolution
    """

    # --- Convert to numpy ---
    def to_numpy(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    # end to_numpy

    theta = to_numpy(theta)
    disp_curve = to_numpy(disp_curve)
    if z_vnoi is not None:
        z_vnoi = to_numpy(z_vnoi)
    # end if

    # --- Handle single model case ---
    if theta.ndim == 1:
        theta = theta[None, :]
    # end if

    if disp_curve.ndim == 1:
        disp_curve = disp_curve[None, :]
    # end if

    if z_vnoi is not None and z_vnoi.ndim == 1:
        z_vnoi = z_vnoi[None, :]
    # end if

    n_models = theta.shape[0]
    periods = np.linspace(p_min, p_max, kmax)

    total_params = theta.shape[1]
    Nmax = (total_params - 2) // 2

    # --- Prepare figure ---
    fig, axes = plt.subplots(
        n_models, 2,
        figsize=(int(7 * fig_size), int(2 * fig_size * n_models)),
        dpi=dpi
    )

    if n_models == 1:
        axes = np.array([axes])
    # end if

    # --- Plot each model ---
    for i in range(n_models):
        n_layers = int(theta[i, 0])
        vpvs = float(theta[i, 1])
        h = theta[i, 2:2 + Nmax][:n_layers]
        vs = theta[i, 2 + Nmax:2 + 2 * Nmax][:n_layers]

        # Compute interfaces
        finite_thicknesses = [t for t in h if t > 0]
        depth_interfaces = np.concatenate(([0], np.cumsum(finite_thicknesses))) if len(finite_thicknesses) > 0 else np.array([0])

        # Déterminer la profondeur max physique :
        # dernier point Voronoï + moitié de l'épaisseur réelle de la dernière couche
        if z_vnoi is not None and np.any(z_vnoi[i] > 0):
            valid_voronoi = z_vnoi[i][z_vnoi[i] > 0.0]
            last_voronoi = valid_voronoi[-1]
            max_depth = last_voronoi + 0.5 * h[-2] if len(h) >= 2 else last_voronoi + 0.5 * h[-1]
        else:
            max_depth = depth_interfaces[-1] + 1.0
        # end if

        # Build step-like profile
        z_plot = [0]
        vs_plot = [vs[0]]
        for j in range(n_layers - 1):
            z_plot.extend([depth_interfaces[j + 1], depth_interfaces[j + 1]])
            vs_plot.extend([vs[j], vs[j + 1]])
        # end for
        z_plot.append(max_depth)
        vs_plot.append(vs[-1])

        # --- Left plot: velocity model ---
        ax_left = axes[i, 0]
        ax_left.plot(z_plot, vs_plot, color="royalblue", linewidth=2)
        ax_left.fill_between(z_plot, 0, vs_plot, color="royalblue", alpha=0.2)
        ax_left.set_xlim(0, max_depth)
        ax_left.set_ylim(min(vs) * 0.9, max(vs) * 1.1)
        ax_left.set_xlabel("Depth [km]")
        ax_left.set_ylabel("Vs [km/s]")
        ax_left.set_title(f"Model {i + 1}: {n_layers} layers (Vp/Vs={vpvs:.2f})")
        ax_left.grid(True, linestyle="--", alpha=0.5)

        # --- Plot Voronoi points (excluding last) ---
        if z_vnoi is not None:
            valid_points = z_vnoi[i][(z_vnoi[i] > 0.0)]
            if len(valid_points) > 0:
                ax_left.scatter(
                    valid_points,
                    np.interp(valid_points, z_plot, vs_plot),
                    color="crimson",
                    s=25,
                    marker="o",
                    label="Voronoi points"
                )
                ax_left.legend(loc="lower right", fontsize=8)
            # end if
        # end if

        # --- Right plot: dispersion curve ---
        ax_right = axes[i, 1]
        ax_right.plot(periods, disp_curve[i], color="tomato", linewidth=2)
        ax_right.set_xlabel("Period [s]")
        ax_right.set_ylabel("Group velocity [km/s]")
        ax_right.set_title("Dispersion curve")
        ax_right.grid(True, linestyle="--", alpha=0.5)
    # end for

    plt.tight_layout()
    plt.show()
# end def plot_velocity_and_dispersion
