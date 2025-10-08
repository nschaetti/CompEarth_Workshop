

import numpy as np
import torch


def sample_prior(
        n_samples=8,
        min_layers: int = 2,
        max_layers=10,
        vpvs: float = 1.75
):
    """
    Sampling model
    """
    samples = []
    for _ in range(n_samples):
        n = np.random.randint(2, max_layers)
        h = np.random.uniform(0.5, 5.0, size=max_layers)
        vs = np.random.uniform(1.5, 4.5, size=max_layers)
        theta = [n, vpvs] + h.tolist() + vs.tolist()
        samples.append(theta)
    # end for
    return torch.tensor(samples, dtype=torch.float32)
# end sample_prior
