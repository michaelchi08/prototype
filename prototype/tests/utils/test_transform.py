import unittest

import numpy as np

from prototype.utils.euler import rotx
from prototype.utils.euler import rotz
from prototype.utils.euler import euler2rot
from prototype.utils.utils import deg2rad
from prototype.utils.transform import Transform
from prototype.utils.transform import T_rdf_flu


class TransformTest(unittest.TestCase):
    def test_transform(self):
        T_WB = Transform("world", "body", t=np.array([1.0, 2.0, 3.0]))
        x = np.array([0.0, 0.0, 0.0, 1.0])

        # print(T_WB * x)
        # print(T_WB * T_WB.data())

    def test_rdf_flu(self):
        x = np.array([1.0, 0.0, 0.0])
        result = T_rdf_flu * x
        result = result.ravel()
        self.assertTrue(np.allclose([0.0, 0.0, 1.0], result))

        x = np.array([0.0, 1.0, 0.0])
        result = T_rdf_flu * x
        result = result.ravel()
        self.assertTrue(np.allclose([-1.0, 0.0, 0.0], result))

        x = np.array([0.0, 0.0, 1.0])
        result = T_rdf_flu * x
        result = result.ravel()
        self.assertTrue(np.allclose([0.0, -1.0, 0.0], result))
