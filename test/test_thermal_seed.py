"""Replay and shape checks for the thermal streams used by SAW validation."""

import numpy as np
import pytest

from mumaxplus import Ferromagnet, Grid, World


def magnet_with_noise(nx):
    world = World((5e-9, 10e-9, 20e-9))
    magnet = Ferromagnet(world, Grid((nx, 1, 1)))
    magnet.msat = 140e3
    magnet.alpha = 5e-4
    magnet.temperature = 300
    return world, magnet


@pytest.mark.parametrize("nx", [1, 2, 3])
def test_seed_replays_odd_and_even_grids(nx):
    world, magnet = magnet_with_noise(nx)
    seed = 2**40 + 2718
    magnet.thermal_seed = seed
    first = magnet.thermal_noise.eval().copy()
    following = magnet.thermal_noise.eval().copy()
    magnet.thermal_seed = seed
    assert magnet.thermal_seed == seed
    np.testing.assert_array_equal(magnet.thermal_noise.eval(), first)
    np.testing.assert_array_equal(magnet.thermal_noise.eval(), following)
    assert first.shape == (3, 1, 1, nx)
    assert np.isfinite(first).all()
    assert np.any(first != following)


def test_seed_changes_stream():
    world, magnet = magnet_with_noise(3)
    magnet.thermal_seed = 1
    first = magnet.thermal_noise.eval().copy()
    magnet.thermal_seed = 2
    assert np.any(magnet.thermal_noise.eval() != first)


def test_nonmagnetic_cells_have_zero_noise():
    world, magnet = magnet_with_noise(3)
    magnet.msat = np.array([[[[140e3, 0., 140e3]]]])
    noise = magnet.thermal_noise.eval()
    assert np.isfinite(noise).all()
    np.testing.assert_array_equal(noise[:, 0, 0, 1], 0.)
    assert np.any(noise[:, 0, 0, 0] != 0.)


def test_zero_temperature_has_zero_noise():
    world, magnet = magnet_with_noise(1)
    magnet.temperature = 0
    np.testing.assert_array_equal(magnet.thermal_noise.eval(), 0.)
