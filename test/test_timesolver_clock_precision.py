"""Clock regressions: late absolute time must not alter autonomous dynamics."""

import numpy as np
import pytest

from mumaxplus import Ferromagnet, Grid, World


def test_many_fixed_steps_preserve_elapsed_time():
    world = World((5e-9, 10e-9, 20e-9))
    solver = world.timesolver
    solver.adaptive_timestep = False
    solver.timestep = 2e-13
    applied_dt = solver.timestep
    solver.steps(500000)
    assert abs(solver.time - 500000 * applied_dt) < 1e-18


def test_repeated_sample_intervals_preserve_elapsed_time():
    world = World((5e-9, 10e-9, 20e-9))
    solver = world.timesolver
    solver.adaptive_timestep = False
    solver.timestep = 2e-13
    for _ in range(20000):
        solver.run(5e-12)
    assert abs(solver.time - 100e-9) < 1e-18


@pytest.mark.parametrize("origin", [0.0, 100e-9])
def test_autonomous_precession_at_late_time(origin):
    world = World((5e-9, 10e-9, 20e-9))
    magnet = Ferromagnet(world, Grid((1, 1, 1)))
    magnet.enable_demag = False
    magnet.msat = 140e3
    magnet.aex = 0.0
    magnet.alpha = 0.0
    magnet.magnetization = (1, 0, 0)
    magnet.bias_magnetic_field = (0, 0, .1)
    world.timesolver.adaptive_timestep = False
    world.timesolver.timestep = 2e-13
    world.timesolver.time = origin
    for _ in range(200):
        world.timesolver.run(5e-12)
    phase = float(magnet.gamma.eval().mean()) * .1 * 1e-9
    expected = np.array([np.cos(phase), np.sin(phase), 0])
    actual = np.array(magnet.magnetization.average())
    assert np.max(abs(actual - expected)) < 2e-5


def test_nonpositive_duration_does_not_advance():
    world = World((5e-9, 10e-9, 20e-9))
    world.timesolver.time = 100e-9
    before = world.timesolver.time
    world.timesolver.run(0)
    world.timesolver.run(-1e-9)
    assert world.timesolver.time == before
