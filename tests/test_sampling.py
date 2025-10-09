import unittest

import numpy as np

from compearth.utils.sampling import sample_models


class TestSampleModels(unittest.TestCase):
    def test_layers_match_thickness_vs_and_voronoi(self) -> None:
        """
        The number of Voronoi points defines the number of layers. Verify that the
        sampled layer count agrees with the populated thicknesses, shear-wave
        velocities, and Voronoi points returned by `sample_models`.
        """
        n_samples = 12
        layers_min = 3
        layers_max = 7
        z_min = 1.0  # avoid zeros so padding checks stay robust
        z_max = 25.0
        vs_min = 1.5
        vs_max = 4.0
        vpvs_fixed = 1.75
        thick_min = 0.75
        rng = np.random.default_rng(20240229)

        theta, z_voronoi = sample_models(
            n_samples=n_samples,
            layers_min=layers_min,
            layers_max=layers_max,
            z_min=z_min,
            z_max=z_max,
            vs_min=vs_min,
            vs_max=vs_max,
            vpvs_fixed=vpvs_fixed,
            thick_min=thick_min,
            random_state=rng,
        )

        self.assertEqual(theta.shape, (n_samples, 2 + 2 * layers_max))
        self.assertEqual(z_voronoi.shape, (n_samples, layers_max))

        theta_np = theta.numpy()
        z_np = z_voronoi.numpy()

        for idx, (sample, z_points) in enumerate(zip(theta_np, z_np)):
            with self.subTest(sample=idx):
                n_layers = int(sample[0])
                self.assertGreaterEqual(n_layers, layers_min)
                self.assertLess(n_layers, layers_max)

                h = sample[2:2 + layers_max]
                vs = sample[2 + layers_max:2 + 2 * layers_max]

                # Vs entries: exactly n_layers entries are populated (> 0), padding stays zero.
                self.assertEqual(np.count_nonzero(vs), n_layers)
                self.assertTrue(np.allclose(vs[n_layers:], 0.0))
                self.assertTrue(np.all(vs[:n_layers] >= vs_min))
                self.assertTrue(np.all(vs[:n_layers] <= vs_max + 0.5))  # extra half-space boost

                # Thickness: first n_layers - 1 must exceed the minimum, last layer is half-space (0 km).
                if n_layers > 1:
                    self.assertTrue(np.all(h[:n_layers - 1] >= thick_min - 1e-6))
                self.assertTrue(np.isclose(h[n_layers - 1], 0.0))
                self.assertTrue(np.allclose(h[n_layers:], 0.0))

                # Voronoi points: one per layer, sorted, padded with zeros afterwards.
                self.assertEqual(np.count_nonzero(z_points), n_layers)
                self.assertTrue(np.all(z_points[:n_layers] >= z_min))
                self.assertTrue(np.all(z_points[:n_layers] <= z_max))
                self.assertTrue(np.all(np.diff(z_points[:n_layers]) >= 0.0))
                self.assertTrue(np.allclose(z_points[n_layers:], 0.0))


if __name__ == "__main__":
    unittest.main()
