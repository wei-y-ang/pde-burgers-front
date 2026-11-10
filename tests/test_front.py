import unittest

import numpy as np

from pde_burgers.front import front_metrics, periodic_distance, track_front


class TestFront(unittest.TestCase):
    def test_periodic_distance(self):
        self.assertAlmostEqual(float(periodic_distance(0.05, 0.95)), 0.10, places=8)

    def test_tracking_moving_sine(self):
        x = np.linspace(0.0, 1.0, 512, endpoint=False)
        t = np.linspace(0.0, 0.5, 51)
        speed_true = 0.4
        # Smooth periodic traveling wave; max |u_x| sits at x = 0.2 + c t.
        phase = 2.0 * np.pi * (x[None, :] - 0.2 - speed_true * t[:, None])
        u = 0.5 * (1.0 + np.sin(phase))

        positions, speed = track_front(u, x, dt=float(t[1] - t[0]))
        self.assertAlmostEqual(float(speed[0]), speed_true, places=2)
        # The sine has two equally steep sides; allow a constant half-period
        # offset, but require the displacement to match the true front speed.
        displacement = positions - positions[0]
        self.assertLess(float(np.max(np.abs(displacement - speed_true * t))), 0.02)

        m = front_metrics(u, u, x, dt=float(t[1] - t[0]))
        self.assertAlmostEqual(m.position_mae, 0.0, places=8)
        self.assertAlmostEqual(m.speed_abs_error, 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
