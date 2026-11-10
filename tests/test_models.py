import unittest

import torch

from pde_burgers.losses import relative_l2
from pde_burgers.models import FNO1d, UNet1d


class TestModels(unittest.TestCase):
    def test_fno_shape(self):
        model = FNO1d(width=8, modes=4, n_layers=2)
        x = torch.randn(2, 1, 256)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 1, 256))

    def test_unet_shape(self):
        model = UNet1d(base=8)
        x = torch.randn(2, 1, 256)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 1, 256))

    def test_relative_l2_zero(self):
        x = torch.randn(3, 1, 64)
        self.assertAlmostEqual(float(relative_l2(x, x)), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
