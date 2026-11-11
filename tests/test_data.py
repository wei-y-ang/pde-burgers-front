import unittest

import numpy as np

from pde_burgers.data import (
    BurgersData,
    BurgersPairs,
    BurgersSequences,
    compute_stats,
    downsample_burgers,
    split_samples,
)


class TestData(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.u = rng.normal(size=(10, 6, 32)).astype(np.float32)

    def test_split_disjoint(self):
        train, val, test = split_samples(10, val_frac=0.2, test_frac=0.2, seed=0)
        self.assertEqual(len(set(train) & set(val)), 0)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(set(val) & set(test)), 0)
        self.assertEqual(len(train) + len(val) + len(test), 10)

    def test_pair_shapes(self):
        mean, std = compute_stats(self.u, np.arange(10))
        ds = BurgersPairs(self.u, np.arange(10), mean, std)
        x, y = ds[0]
        self.assertEqual(tuple(x.shape), (1, 32))
        self.assertEqual(tuple(y.shape), (1, 32))

    def test_sequence_shapes(self):
        mean, std = compute_stats(self.u, np.arange(10))
        ds = BurgersSequences(self.u, np.arange(10), mean, std, horizon=3)
        x, y = ds[0]
        self.assertEqual(tuple(x.shape), (1, 32))
        self.assertEqual(tuple(y.shape), (3, 32))

    def test_downsample_uses_shared_points(self):
        x = (np.arange(32) + 0.5).astype(np.float32) / 32
        t = np.arange(6, dtype=np.float32)
        data = BurgersData(self.u, x, t, nu=0.01)
        coarse = downsample_burgers(data, 16)
        self.assertEqual(coarse.u.shape, (10, 6, 16))
        np.testing.assert_array_equal(coarse.u, self.u[:, :, ::2])
        np.testing.assert_array_equal(coarse.x, x[::2])
        np.testing.assert_array_equal(coarse.nu_values, data.nu_values)

    def test_downsample_rejects_incompatible_resolution(self):
        x = (np.arange(32) + 0.5).astype(np.float32) / 32
        data = BurgersData(self.u, x, np.arange(6, dtype=np.float32), nu=0.01)
        with self.assertRaises(ValueError):
            downsample_burgers(data, 17)


if __name__ == "__main__":
    unittest.main()
