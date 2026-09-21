import numpy as np
from mumaxplus import NcAfm, Grid, World

def max_absolute_error(result, wanted):
    """Maximum error for vector quantities."""
    return np.max(np.linalg.norm(result - wanted, axis=0))

def max_semirelative_error(result, wanted):
    """Like relative error, but divides by the maximum of wanted.
    Useful when removing units but the results go through zero.
    """
    return max_absolute_error(result, wanted) / np.max(abs(wanted))

def compute_octupole_vector(m1, m2, m3, ms1, ms2, ms3):

    def rotate(m, angle):
        return np.array([np.cos(angle) * m[0] - np.sin(angle) * m[1],
                         np.sin(angle) * m[0] + np.cos(angle) * m[1], m[2]])

    numerator = (m1 * ms1 + rotate(m2, -2 * np.pi / 3) * ms2 +
                 rotate(m3, -4 * np.pi / 3) * ms3)
    total = ms1 + ms2 + ms3
    return np.divide(numerator, total, out=np.zeros_like(numerator), where=total != 0)

class TestOctupoleVector:
    def test_octupole_vector(self):
        world = World((1, 1, 1))
        magnet = NcAfm(world, Grid((32, 32, 32)))
        ms1, ms2, ms3 = 123, 456, 789
        magnet.sub1.msat = ms1
        magnet.sub2.msat = ms2
        magnet.sub3.msat = ms3
        m1 = magnet.sub1.magnetization()
        m2 = magnet.sub2.magnetization()
        m3 = magnet.sub3.magnetization()

        result = magnet.octupole_vector()
        wanted = compute_octupole_vector(m1, m2, m3, ms1, ms2, ms3)
        assert max_semirelative_error(result, wanted) < 5e-6

    def test_octupole_vector_uniform(self):
        world = World((1, 1, 1))
        magnet = NcAfm(world, Grid((1, 1, 1)))
        magnet.msat = 10
        magnet.magnetization = (1, 0, 0)

        result = magnet.octupole_vector()
        assert np.allclose(result, 0, atol=1e-6)

    def test_octupole_vector_120(self):
        world = World((1, 1, 1))
        magnet = NcAfm(world, Grid((1, 1, 1)))
        magnet.msat = 10
        for i, sub in enumerate(magnet.sublattices):
            theta = i * 120 * np.pi / 180
            sub.magnetization = (np.cos(theta), np.sin(theta), 0)

        result = magnet.octupole_vector()
        assert np.allclose(result.squeeze(), [1, 0, 0])

    def test_reversed_index_chirality(self):
        world = World((1, 1, 1))
        magnet = NcAfm(world, Grid((1, 1, 1)))
        magnet.msat = 10
        for i, sub in enumerate(magnet.sublattices):
            theta = -i * 2 * np.pi / 3
            sub.magnetization = (np.cos(theta), np.sin(theta), 0)
        assert np.allclose(magnet.octupole_vector(), 0, atol=1e-6)

    def test_local_zero_ms(self):
        world = World((1, 1, 1))
        magnet = NcAfm(world, Grid((2, 1, 1)))
        magnet.msat = np.array([[[[10.0, 0.0]]]])
        magnet.magnetization = (0, 0, 1)
        result = magnet.octupole_vector()
        assert np.isfinite(result).all()
        assert np.allclose(result[:, 0, 0, 0], [0, 0, 1])
        assert np.allclose(result[:, 0, 0, 1], 0)
