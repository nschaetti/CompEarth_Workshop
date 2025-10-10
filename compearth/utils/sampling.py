


from typing import Union
import numpy as np
import torch
from typing import Tuple
import ruptures as rpt


def sample_models(
    n_samples: int = 8,
    layers_min: int = 2,
    layers_max: int = 10,
    z_min: float = 0.0,
    z_max: float = 60.0,
    vs_min: float = 1.5,
    vs_max: float = 4.5,
    vpvs_fixed: float = 1.75,
    thick_min: float = 0.5,
    sort_vs: bool = False,
    random_state: np.random.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Correct version:
    - N Voronoi points = N layers
    - Last interface = z_max
    - Last layer already acts as the half-space (no extra layer)
    """
    if random_state is None:
        random_state = np.random.default_rng()
    # end if

    samples, z_vnoi_all = [], []

    for _ in range(n_samples):
        # --- 1. Number of Voronoi points = number of layers ---
        n_layers = random_state.integers(layers_min, layers_max)

        # --- 2. Generate valid Voronoi midpoints ---
        valid = False
        while not valid:
            z_vnoi = np.sort(random_state.uniform(low=z_min, high=z_max, size=n_layers))

            # Interfaces halfway between consecutive midpoints
            interfaces = np.zeros(n_layers + 1)
            interfaces[0] = z_min
            interfaces[1:-1] = 0.5 * (z_vnoi[1:] + z_vnoi[:-1])
            interfaces[-1] = z_max

            h = np.diff(interfaces)
            h[-1] = 0.0

            if np.all(h[:-1] >= thick_min):
                valid = True
            # end if
        # end while

        # --- 3. Sample Vs values ---
        vs = random_state.uniform(low=vs_min, high=vs_max, size=n_layers)
        if sort_vs:
            vs = np.sort(vs)
        # end if

        # Increase last Vs slightly (half-space behaviour)
        vs[-1] += random_state.uniform(0.2, 0.5)

        # --- 4. Pad to layers_max ---
        h_padded = np.zeros(layers_max)
        vs_padded = np.zeros(layers_max)
        z_padded = np.zeros(layers_max)

        h_padded[:n_layers] = h
        vs_padded[:n_layers] = vs
        z_padded[:n_layers] = z_vnoi

        # --- 5. Assemble θ vector ---
        theta = [n_layers, vpvs_fixed] + h_padded.tolist() + vs_padded.tolist()
        samples.append(theta)
        z_vnoi_all.append(z_padded)
    # end for

    theta = torch.tensor(samples, dtype=torch.float32)
    z_vnoi = torch.tensor(z_vnoi_all, dtype=torch.float32)

    return theta, z_vnoi
# end def sample_models


def theta_to_velocity_profile(
        theta: Union[np.ndarray, torch.Tensor],
        depth_max: float = 60.0,
        n_points: int = 60
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a layered model θ into a sampled velocity profile Vs(z).

    Parameters
    ----------
    theta : np.ndarray or torch.Tensor
        Model parameters [n_layers, vpvs, h_1...h_Nmax, vs_1...vs_Nmax].
    depth_max : float
        Maximum depth in km for sampling.
    n_points : int
        Number of depth samples for the velocity profile.

    Returns
    -------
    depth : np.ndarray
        Depth samples (km).
    vs_profile : np.ndarray
        Corresponding shear-wave velocities (km/s).
    """
    if isinstance(theta, torch.Tensor):
        theta = theta.detach().cpu().numpy()
    # end if

    n_layers = int(theta[0])
    vpvs = float(theta[1])
    Nmax = (len(theta) - 2) // 2
    h = theta[2:2 + Nmax][:n_layers]
    vs = theta[2 + Nmax:2 + 2 * Nmax][:n_layers]

    # Compute cumulative depth for interfaces
    layer_tops = np.concatenate(([0], np.cumsum(h[:-1])))
    layer_bottoms = np.concatenate((np.cumsum(h[:-1]), [depth_max]))
    depth = np.linspace(0, depth_max, n_points)

    vs_profile = np.zeros_like(depth)
    for i in range(n_layers):
        z_top = layer_tops[i]
        if i == n_layers - 1:
            # Last layer → half-space (extends to infinity)
            mask = depth >= z_top
        else:
            z_bottom = layer_bottoms[i]
            mask = (depth >= z_top) & (depth < z_bottom)
        # end if
        vs_profile[mask] = vs[i]
    # end for

    return depth, vs_profile
# end def theta_to_velocity_profile


def flatten_models(
        samples: torch.Tensor,
        penalty: float = 0.0,
):
    """
    Flatten models.
    """
    assert samples.ndim == 2

    def flatten(vs, pen):
        algo = rpt.Pelt(model="l2").fit(vs)
        bkps = algo.predict(pen=pen)
        vs_flat = np.zeros_like(vs)
        start = 0
        for end in bkps:
            vs_flat[start:end] = np.mean(vs[start:end])
            start = end
        # end for
        return vs_flat
    # end flatten

    flat_models = list()
    for b in range(samples.shape[0]):
        m_flat = flatten(
            samples[b].cpu().numpy(),
            pen=penalty
        )
        flat_models.append(torch.tensor(m_flat).unsqueeze(0))
    # end for

    return torch.cat(flat_models, dim=0)
# end def flatten_models
