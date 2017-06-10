import unittest
import math

import numpy as np

from prototype.utils.math import deg2rad
from prototype.utils.math import rad2deg
from prototype.utils.math import quat2rot
from prototype.utils.math import quat2euler
from prototype.utils.math import euler2rot
from prototype.utils.math import euler2quat


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
        pass

    def test_euler2rot(self):
        euler = [0.1, 0.2, 0.3]

        # test euler-sequence 123
        R1 = euler2rot(euler, 123)
        quat = euler2quat(euler, 123)
        R2 = quat2rot(quat)
        self.assertTrue(np.allclose(R1, R2))

        # test euler-sequence 321
        R = euler2rot(euler, 321)
        expected = [[0.93629336, -0.27509585, 0.21835066],
                    [0.28962948, 0.95642509, -0.03695701],
                    [-0.19866933, 0.0978434, 0.97517033]]
        self.assertTrue(np.allclose(expected, R))

    def test_euler2quat(self):
        euler = [0.1, 0.2, 0.3]

        # test euler-sequence 123
        quat = euler2quat(euler, 123)
        expected = [0.981856172866081,
                    0.06407134770607117,
                    0.09115754934299072,
                    0.15343930202422262]
        self.assertEqual(expected, quat)

        # test euler-sequence 321
        quat = euler2quat(euler, 321)
        expected = [0.9836907551963402,
                    0.03428276336964735,
                    0.10605752555507605,
                    0.14117008022286104]
        self.assertEqual(expected, quat)

    def test_quat2euler(self):
        expected = np.array([0.1, 0.2, 0.3])

        # test quaternion to euler 123
        q = [0.981856172866081,
             0.06407134770607117,
             0.09115754934299072,
             0.15343930202422262]
        euler = quat2euler(q, 123)
        self.assertTrue(np.allclose(expected, euler))

        # test quaternion to euler 321
        q = [0.9836907551963402,
             0.03428276336964735,
             0.10605752555507605,
             0.14117008022286104]
        euler = quat2euler(q, 321)
        self.assertTrue(np.allclose(expected, euler, rtol=1e-01))

    def test_quat2rot(self):
        # import numpy as np
        pass

        # euler = [0.1, 0.2, 0.3]

        # R1 = euler2rot(euler, 123)
        # x = np.dot(R1, np.array([1.0, 0.0, 0.0]))

        # print(x)
        # q = euler2quat(euler, 123)
        # R2 = quat2rot(q)
        # print()
        # print(R1)
        # print()

        # import numpy as np
        # print(np.dot(R1, vector))
        # print(np.dot(R2, vector))

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
