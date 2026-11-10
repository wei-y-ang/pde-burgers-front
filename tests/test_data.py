import unittest

import numpy as np

from pde_burgers.data import BurgersPairs, BurgersSequences, compute_stats, split_samples


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


if __name__ == "__main__":
    unittest.main()
