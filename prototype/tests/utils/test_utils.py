import unittest
import math

import numpy as np

from prototype.utils.utils import rotx
from prototype.utils.utils import roty
from prototype.utils.utils import rotz
from prototype.utils.utils import deg2rad
from prototype.utils.utils import rad2deg
from prototype.utils.quaternion.hamiltonian import quat2rot
from prototype.utils.euler import euler2rot


class MathTest(unittest.TestCase):
    def test_deg2rad(self):
        r = deg2rad(360)
        self.assertTrue((r - 2 * math.pi) < 0.0001)

    def test_rad2deg(self):
        d = rad2deg(math.pi)
        self.assertTrue((d - 180.0) < 0.0001)

    def test_rotx(self):
        pass

    def test_roty(self):
        pass

    def test_rotz(self):
        theta = 0.785
        R = rotz(theta)
        p = np.dot(R, [2, 1, 0])
        expected = [0.70795136, 2.12103863, 0.0]
        self.assertTrue(np.allclose(expected, p))

    def test_enu2nwu(self):
        pass

    def test_cf2nwu(self):
        pass

    def test_cf2enu(self):
        pass

    def test_ned2enu(self):
        pass

    def test_nwu2enu(self):
        pass

    def test_nwu2edn(self):
        pass

    def test_wrap180(self):
        pass

    def test_wrap360(self):
        pass
