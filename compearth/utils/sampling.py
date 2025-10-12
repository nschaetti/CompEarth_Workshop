


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
        rng: np.random.Generator | None = None,
        seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Correct version:
    - N Voronoi points = N layers
    - Last interface = z_max
    - Last layer already acts as the half-space (no extra layer)
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    # end if

    samples, z_vnoi_all = [], []

    for _ in range(n_samples):
        # --- 1. Number of Voronoi points = number of layers ---
        n_layers = rng.integers(layers_min, layers_max)

        # --- 2. Generate valid Voronoi midpoints ---
        valid = False
        while not valid:
            z_vnoi = np.sort(rng.uniform(low=z_min, high=z_max, size=n_layers))

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
        vs = rng.uniform(low=vs_min, high=vs_max, size=n_layers)
        if sort_vs:
            vs = np.sort(vs)
        # end if

        # Increase last Vs slightly (half-space behaviour)
        vs[-1] += rng.uniform(0.2, 0.5)

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


def posterior_sample_to_theta(
        z: Union[np.ndarray, torch.Tensor],
        vs_batch: Union[np.ndarray, torch.Tensor],
        vpvs: float = 1.75,
        penalty_min: float = 0.1,
        penalty_max: float = 5.0,
        model: str = "l2",
        max_layers: int = 20,
        rng: np.random.RandomState = None,
        seed: int = 42,
) -> Tuple[torch.Tensor, list, np.ndarray]:
    """
    Convert multiple posterior velocity samples into layered Earth models (θ),
    using PELT segmentation with a random penalty drawn uniformly for each sample.
    Breakpoints are returned in depth (km) rather than indices.

    Parameters
    ----------
    z : np.ndarray or torch.Tensor
        Depth coordinates (D_z,), in km.
    vs_batch : np.ndarray or torch.Tensor
        Velocity profiles (N, D_z)
    vpvs : float
        Fixed Vp/Vs ratio for all models
    penalty_min, penalty_max : float
        Range for random penalties
    model : str
        Cost model for ruptures (default: 'l2')
    max_layers : int
        Maximum number of layers (for padding in θ)
    rng : np.random.RandomState
        Random number generator
    seed : int
        Random seed for reproducibility

    Returns
    -------
    theta_all : torch.Tensor
        Model parameters of shape (N, 2 + 2 * max_layers)
        [n_layers, vpvs, h_1...h_max, vs_1...vs_max]
    bkps_all_km : list[list[float]]
        List of breakpoint depths (km) for each sample
    penalties : np.ndarray
        Penalties drawn for each sample
    """
    # --- Convert to numpy ---
    if hasattr(vs_batch, "detach"):
        vs_batch = vs_batch.detach().cpu().numpy()
    # end if

    if hasattr(z, "detach"):
        z = z.detach().cpu().numpy()
    # end if

    n_samples, n_depths = vs_batch.shape
    theta_all = []
    bkps_all_km = []
    penalties = []

    if rng is None:
        rng = np.random.default_rng(seed)
    # end if

    for i in range(n_samples):
        vs = vs_batch[i]
        penalty = rng.uniform(penalty_min, penalty_max)
        penalties.append(penalty)

        algo = rpt.Pelt(model=model).fit(vs)
        bkps = algo.predict(pen=penalty)

        # --- Convert breakpoints (indices) -> depths (km)
        bkps_depth = [z[min(end - 1, len(z) - 1)] for end in bkps if end <= len(z)]
        bkps_all_km.append(bkps_depth)

        # --- Build layers ---
        vs_layers = []
        h_layers = []
        start = 0

        for end in bkps:
            segment_vs = vs[start:end]
            segment_z = z[start:end]
            mean_vs = np.mean(segment_vs)
            vs_layers.append(mean_vs)
            if len(segment_z) > 0:
                h_layers.append(segment_z[-1] - segment_z[0])
            else:
                h_layers.append(0.0)
            # end if
            start = end
        # end for

        # Half-space (last layer)
        if h_layers:
            h_layers[-1] = 0.0
        # end if

        # Padding
        n_layers = len(vs_layers)
        h_padded = np.zeros(max_layers)
        vs_padded = np.zeros(max_layers)
        h_padded[:n_layers] = h_layers[:max_layers]
        vs_padded[:n_layers] = vs_layers[:max_layers]

        # Assemble θ vector
        theta = [n_layers, vpvs] + h_padded.tolist() + vs_padded.tolist()
        theta_all.append(theta)
    # end for

    theta_all = torch.tensor(theta_all, dtype=torch.float32)
    penalties = np.array(penalties)

    return theta_all, bkps_all_km, penalties
# end def posterior_sample_to_theta


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


def random_flatten_models(
        samples: Union[np.ndarray, torch.Tensor],
        penalty_min: float = 0.1,
        penalty_max: float = 5.0,
        model: str = "l2",
        rng: np.random.RandomState = None,
        seed: int = 42,
) -> np.ndarray:
    """
    Flatten posterior samples using PELT segmentation, with a random penalty
    drawn uniformly between `penalty_min` and `penalty_max` for each sample.

    Parameters
    ----------
    samples : np.ndarray or torch.Tensor
        Posterior samples of shape (N, D_z)
    penalty_min, penalty_max : float
        Range of random penalties for the PELT algorithm.
    model : str
        Cost model for ruptures (default: "l2").
    rng : np.random.RandomState
        Random seed for reproducibility
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    vs_flat_all : np.ndarray
        Flattened velocity profiles of shape (N, D_z)
    """
    # --- Convert to numpy ---
    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().numpy()

    n_samples, depth_points = samples.shape
    vs_flat_all = np.zeros_like(samples)

    if rng is None:
        rng = np.random.default_rng(seed)
    # end if

    # --- Flatten each sample with a random penalty ---
    for i in range(n_samples):
        s = samples[i]
        penalty = rng.uniform(penalty_min, penalty_max)

        algo = rpt.Pelt(model=model).fit(s)
        bkps = algo.predict(pen=penalty)

        vs_flat = np.zeros_like(s)
        start = 0
        for end in bkps:
            vs_flat[start:end] = np.mean(s[start:end])
            start = end
        # end for

        vs_flat_all[i] = vs_flat
    # end for

    return vs_flat_all
# end def random_flatten_models



