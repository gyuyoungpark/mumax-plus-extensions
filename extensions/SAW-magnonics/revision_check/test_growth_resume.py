"""Focused regressions for the interrupted early-growth correction (CPU only)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import growth_interval as GI


def two_branch(t, a0=1e-4):
    return a0 * np.exp(3e8 * np.minimum(t, 4e-9)
                       - 1.2e8 * np.maximum(t - 4e-9, 0))


class EarlyGrowthTests(unittest.TestCase):
    def test_longer_acquisition_does_not_hide_the_early_map_interval(self):
        dt, nx, dx, f0 = 20e-12, 128, 5e-9, 3e9
        k = 8 * 2 * np.pi / (nx * dx)
        x = (np.arange(nx) + 0.5) * dx
        sizes = []
        for duration in (30e-9, 120e-9):
            t = np.arange(round(duration / dt) + 1) * dt
            my = two_branch(t)[:, None] * np.cos(
                k * x[None, :] - 2 * np.pi * f0 * t[:, None])
            k_ax, gamma, _, status, _, _, _, meta = GI.gamma_map(my, dx, dt, f0)
            j = int(np.argmin(np.abs(k_ax - k)))
            self.assertFalse(status[j] == GI.OK and gamma[j] < 0,
                             (duration, status[j], gamma[j]))
            sizes.append(meta["seg"])
        self.assertEqual(sizes[0], sizes[1])

    def test_saturation_reentry_cannot_supply_negative_bracket_point(self):
        t = np.arange(1501) * 20e-12
        for a0 in (0.02, 0.03, 0.05, 0.06):
            with self.subTest(a0=a0):
                r = GI.fit_growth_interval(t, two_branch(t, a0), sat_level=0.1)
                self.assertEqual(r["status"], GI.NOT_SUMMARISABLE, r)
                self.assertTrue(np.isnan(r["gamma"]))
                self.assertAlmostEqual(r["late"]["gamma"] / -1.2e8, 1, places=6)
                positive = GI.fit_growth_interval(t, two_branch(t))
                bracket = GI.bracket_from_fits([
                    GI.point_from_fit(3e-5, r),
                    GI.point_from_fit(7e-5, positive),
                ])
                self.assertEqual(bracket["status"], "NOT_EVALUABLE")

    def test_record_linearity_breach_is_not_reopened(self):
        t = np.arange(1501) * 20e-12
        mask = (t < 1e-9) | (t > 10e-9)
        r = GI.fit_growth_interval(t, two_branch(t), linear_mask=mask)
        self.assertEqual(r["status"], GI.NOT_SUMMARISABLE, r)
        self.assertTrue(np.isnan(r["gamma"]))
        self.assertAlmostEqual(r["late"]["gamma"] / -1.2e8, 1, places=6)

    def test_readable_early_growth_survives_saturation_and_long_decay(self):
        t = np.arange(1501) * 20e-12
        r = GI.fit_growth_interval(t, two_branch(t, 0.014), sat_level=0.08)
        self.assertEqual(r["status"], GI.OK, r)
        self.assertAlmostEqual(r["gamma"] / 3e8, 1, places=6)
        self.assertIsNotNone(r["late"])
        self.assertAlmostEqual(r["late"]["gamma"] / -1.2e8, 1, places=6)
        self.assertLess(r["t1"], 4e-9)

    def test_refused_later_window_retains_its_diagnostic(self):
        t = np.arange(2001) * 20e-12
        a = 1e-4 * np.exp(3e8 * np.minimum(t, 0.5e-9)
                          - 1.2e8 * np.maximum(t - 0.5e-9, 0))
        a[(t > 3e-9) & (t < 4e-9)] = 1e-9
        r = GI.fit_growth_interval(t, a)
        self.assertEqual(r["status"], GI.NOT_SUMMARISABLE)
        self.assertTrue(np.isnan(r["gamma"]))
        self.assertIsNotNone(r["late"])
        self.assertAlmostEqual(r["late"]["gamma"] / -1.2e8, 1, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
