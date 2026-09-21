"""Native uniform-Zeeman cavity regression tests (requires DOUBLE and a GPU).

These checks compare the fresh engine to a nonlinear energy-derived CPU ODE,
not to stored simulation data. Run with MUMAXPLUS_FP_PRECISION=DOUBLE.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from mumaxplus import Antiferromagnet, FP_PRECISION, Grid, World

pytestmark = pytest.mark.skipif(FP_PRECISION != "DOUBLE", reason="precision regression uses DOUBLE")

GAMMA = 1.7595e11
BE, BA, MS, LAT = 54.96244105526874, 0.7865384610879607, 624000.0, 3.13e-10


def configured(auxiliary=False):
    world = World((40e-9, 40e-9, 10e-9))
    afm = Antiferromagnet(world, Grid((1, 1, 1)))
    for sub in afm.sublattices:
        sub.gamma = GAMMA
        sub.msat = MS
        sub.ku1 = BA * MS / 2
        sub.anisU = (0, 0, 1)
        sub.alpha = 0.003
        sub.aex = 0
        sub.enable_demag = False
        sub.enable_openbc = True
    afm.afmex_cell = -BE * MS * LAT**2 / 4
    afm.afmex_nn = 0
    afm.latcon = LAT
    m1 = np.array([2e-6, -1e-6, 1.0]); m1 /= np.linalg.norm(m1)
    m2 = np.array([1e-6, 3e-6, -1.0]); m2 /= np.linalg.norm(m2)
    afm.sub1.magnetization = tuple(m1)
    afm.sub2.magnetization = tuple(m2)
    c = afm._impl
    c.cavity_omega = 2 * np.pi * 263e9
    c.cavity_kappa = 2 * np.pi * 0.3e9
    c.cavity_h0 = 0.4
    c.cavity_energy_field = BE
    c.aux_omega = 2 * np.pi * 269e9
    c.aux_kappa = 2 * np.pi * 0.7e9
    c.aux_j = 2 * np.pi * 1.2e9
    c.aux_h0 = 0.15
    c.enable_aux_mode = auxiliary
    c.enable_cavity_afm = True
    c.cavity_amplitude().set((1e-5, -3e-6, 0))
    c.aux_amplitude().set((7e-6, 2e-6, 0))
    world._impl.reset_timesolver_equations()
    world.timesolver.adaptive_timestep = False
    world.timesolver.timestep = 5e-15
    return world, afm


def state(afm):
    c = afm._impl
    return np.r_[afm.sub1.magnetization.average(), afm.sub2.magnetization.average(),
                 c.cavity_amplitude().eval().reshape(-1)[:2],
                 c.aux_amplitude().eval().reshape(-1)[:2]]


def test_uniform_field_enters_both_gilbert_torques():
    _, afm = configured(auxiliary=True)
    c = afm._impl
    c.enable_cavity_afm = False
    before_h = [np.asarray(s.effective_field.average()) for s in afm.sublattices]
    before_t = [np.asarray(s.torque.average()) for s in afm.sublattices]
    c.enable_cavity_afm = True
    h = np.array([0.4 * 1e-5 + 0.15 * 7e-6, 0, 0])
    for i, sub in enumerate(afm.sublattices):
        np.testing.assert_allclose(np.asarray(sub.effective_field.average()) - before_h[i], h,
                                   rtol=1e-10, atol=2e-14)
        m = np.asarray(sub.magnetization.average())
        expected = -GAMMA / (1 + 0.003**2) * (
            np.cross(m, h) + 0.003 * np.cross(m, np.cross(m, h)))
        np.testing.assert_allclose(np.asarray(sub.torque.average()) - before_t[i], expected,
                                   rtol=2e-10, atol=2e-4)


@pytest.mark.parametrize("auxiliary", [False, True])
def test_nonlinear_reciprocal_dynamics(auxiliary):
    world, afm = configured(auxiliary)
    c = afm._impl
    initial = state(afm)
    wa, wd, ka, kd, j = c.cavity_omega, c.aux_omega, c.cavity_kappa, c.aux_kappa, c.aux_j
    ha, hd = c.cavity_h0, c.aux_h0

    def rhs(t, x):
        m1, m2 = x[:3], x[3:6]
        q, p, d, pd = x[6:]
        bx = ha * q + (hd * d if auxiliary else 0)
        out = []
        for m, other in ((m1, m2), (m2, m1)):
            h = -BE * other + np.array([bx, 0, BA * m[2]])
            mxh = np.cross(m, h)
            out.extend(-GAMMA / (1 + 0.003**2) * (mxh + 0.003 * np.cross(m, mxh)))
        mx = m1[0] + m2[0]
        out.extend([wa * p, -wa * q - 2 * ka * p + wa * ha * mx / (2 * BE)
                    - (2 * j * np.sqrt(wa / wd) * d if auxiliary else 0)])
        out.extend([wd * pd, -wd * d - 2 * kd * pd + wd * hd * mx / (2 * BE)
                    - 2 * j * np.sqrt(wd / wa) * q] if auxiliary else [0, 0])
        return out

    duration = 1e-12
    reference = solve_ivp(rhs, (0, duration), initial, method="DOP853",
                          rtol=2e-12, atol=1e-17, max_step=5e-15)
    assert reference.success
    world.timesolver.steps(200)
    # All spin and oscillator channels are compared, including both reciprocal
    # auxiliary factors and damping in the full (not rotating-wave) dynamics.
    np.testing.assert_allclose(state(afm), reference.y[:, -1], rtol=2e-8, atol=2e-13)
