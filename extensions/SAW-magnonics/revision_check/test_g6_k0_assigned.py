"""The three assigned G6/k0 failures; analytic CPU fixtures only."""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "runs")]
import G6_clean_seed as G6
import G4_commensurate_onset as G4
import saw_analysis as SA
from test_adversarial_p1_verify import _wave, _write_g6


class G6AssignedTests(unittest.TestCase):
    def setUp(self):
        self.nt = 256
        self.control = _wave(self.nt, G6.BOX["NX"], G6.BOX["dx"], G6.DT_REC,
                             G6.K_HALF, G6.F_SAW / 2, 1e-3 * np.ones(self.nt))

    def rows(self, records):
        with tempfile.TemporaryDirectory(prefix="g6_assigned_") as tmp:
            with patch.object(G6, "CKPT", tmp):
                for name, my, eps, seed, overrides in records:
                    _write_g6(os.path.join(tmp, name), my, 0.0, eps,
                              "T0_phys", seed, **overrides)
                with redirect_stdout(io.StringIO()):
                    return G6.analyze()

    def test_unknown_seed_records_are_individually_unreadable(self):
        for seed in (None, float("nan"), float("inf"), -float("inf")):
            with self.subTest(seed=seed):
                rows = self.rows([
                    ("G6_T0_phys_eps1e-04_r0.npz", 2*self.control, G6.EPS_WORK, seed, {}),
                    ("G6_T0_phys_eps1e-04_r1.npz", .04*self.control, G6.EPS_WORK, seed, {}),
                    ("G6_T0_phys_eps0e+00_r0.npz", self.control, 0.0, seed, {}),
                ])
                pumped = [r for r in rows if r["eps0"] == G6.EPS_WORK]
                self.assertEqual(len(pumped), 2)
                for row in pumped:
                    self.assertEqual(row["status"], "NOT_DETERMINABLE")
                    self.assertEqual(row["n"], 1)
                    self.assertTrue(np.isnan(row["ratio_mean"]))
                    self.assertIn("a_seed", row["reason"])

    def test_invalid_control_cannot_supply_a_ratio(self):
        bad_controls = [
            ("zero", np.zeros_like(self.control), {}),
            ("nan", np.full_like(self.control, np.nan), {}),
            ("inf", np.full_like(self.control, np.inf), {}),
            ("unfinished", self.control, {"done": False}),
            ("unknown completion", self.control, {"done": "unknown"}),
            ("partial trace", self.control, {"nt_total": self.nt+1}),
            ("partial index", self.control, {"next_index": self.nt-1}),
        ]
        for label, control, overrides in bad_controls:
            with self.subTest(label=label):
                rows = self.rows([
                    ("G6_T0_phys_eps1e-04_r0.npz", 2*self.control, G6.EPS_WORK, 1e-3, {}),
                    ("G6_T0_phys_eps0e+00_r0.npz", control, 0.0, 1e-3, overrides),
                ])
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["status"], "NOT_DETERMINABLE")
                self.assertTrue(np.isnan(rows[0]["ratio_mean"]))
                self.assertIn("control", rows[0]["reason"])

    def test_partial_control_is_not_hidden_by_a_valid_repeat(self):
        rows = self.rows([
            ("G6_T0_phys_eps1e-04_r0.npz", 2*self.control, G6.EPS_WORK, 1e-3, {}),
            ("G6_T0_phys_eps0e+00_r0.npz", self.control, 0.0, 1e-3, {}),
            ("G6_T0_phys_eps0e+00_r1.npz", self.control, 0.0, 1e-3, {"done": False}),
        ])
        self.assertEqual(rows[0]["status"], "NOT_DETERMINABLE")
        self.assertIn("r1.npz", rows[0]["reason"])

    def test_valid_control_still_measures_the_declared_gain(self):
        rows = self.rows([
            ("G6_T0_phys_eps1e-04_r0.npz", 2*self.control, G6.EPS_WORK, 1e-3, {}),
            ("G6_T0_phys_eps0e+00_r0.npz", self.control, 0.0, 1e-3, {}),
        ])
        self.assertEqual(rows[0]["status"], "ok")
        self.assertAlmostEqual(rows[0]["ratio_mean"], 4.0, places=6)


class K0AssignedTests(unittest.TestCase):
    def test_power_requires_known_uniform_initial_diagnostic(self):
        nt, nx, dx, dt = 2001, G4.BOX["NX"], G4.BOX["dx"], G4.DT_REC_DISP
        f0 = 3e9
        # Uniform and off-grid components share a frequency. Spectral overlap
        # is not a reason to reject the deliberately injected uniform mode.
        uniform = _wave(nt, nx, dx, dt, 0.0, f0,
                        G4.K0_DIAG_AMP * np.ones(nt))
        offgrid = _wave(nt, nx, dx, dt, 6.5*2*np.pi/(nx*dx), f0,
                        1e-4*np.ones(nt))
        ka, fa, spectrum = SA.spectrum(uniform+offgrid, dx, dt, window="hann")
        for missing in (None, 0.0, np.nan, np.inf):
            with self.subTest(missing=missing):
                result = G4.k0_signal_check(ka, fa, spectrum, 2*G4.Q,
                                            uniform_initial_amplitude=missing)
                self.assertEqual(result["status"], "NOT_EVALUABLE")
                self.assertIn("origin", result["reason"])
        valid = G4.k0_signal_check(ka, fa, spectrum, 2*G4.Q,
                                   uniform_initial_amplitude=G4.K0_DIAG_AMP)
        self.assertEqual(valid["status"], "ok", valid["reason"])
        self.assertTrue(valid["uniform_initial_known"])

    def test_known_injection_still_needs_a_readable_signal(self):
        k = np.array([-2.0, 0.0, 2.0])
        f = np.array([0.0, 1e9, 2e9])
        result = G4.k0_signal_check(k, f, np.zeros((3, 3)), 1.0,
                                    uniform_initial_amplitude=G4.K0_DIAG_AMP)
        self.assertEqual(result["status"], "NOT_EVALUABLE")
        self.assertNotIn("no uniform excitation", result["reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
