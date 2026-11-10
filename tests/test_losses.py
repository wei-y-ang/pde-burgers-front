import unittest

import torch

from pde_burgers.losses import burgers_residual_step, relative_l2


class TestLosses(unittest.TestCase):
    def test_residual_shape(self):
        u_t = torch.randn(4, 64)
        u_next = torch.randn(4, 64)
        r = burgers_residual_step(u_t, u_next, dx=0.01, dt=0.01, nu=0.01)
        self.assertEqual(tuple(r.shape), (4, 62))

    def test_residual_zero_for_constant(self):
        u_t = torch.ones(2, 32)
        u_next = torch.ones(2, 32)
        r = burgers_residual_step(u_t, u_next, dx=0.01, dt=0.01, nu=0.01)
        self.assertLess(float(r.abs().max()), 1e-6)

    def test_relative_l2(self):
        x = torch.randn(3, 1, 16)
        self.assertAlmostEqual(float(relative_l2(x, x)), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
