import unittest

import numpy as np

from prototype.utils.euler import rotx
from prototype.utils.euler import rotz
from prototype.utils.euler import euler2rot
from prototype.utils.utils import deg2rad
from prototype.utils.transform import Transform


class TransformTest(unittest.TestCase):
    def test_transform(self):
        T_WB = Transform("world", "body", t=np.array([1.0, 2.0, 3.0]))
        x = np.array([0.0, 0.0, 0.0, 1.0]).reshape((4, 1))

        print(T_WB * x)
        print(T_WB * T_WB.data())
