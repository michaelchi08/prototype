import unittest

import numpy as np


from prototype.utils.linalg import skew
from prototype.utils.linalg import nullspace


class LinalgTest(unittest.TestCase):
    def test_skew(self):
        v = np.array([[1.0], [2.0], [3.0]])
        X = skew(v)
        X_expected = np.array([[0.0, -3.0, 2.0],
                               [3.0, 0.0, -1.0],
                               [-2.0, 1.0, 0.0]])

        self.assertTrue(np.array_equal(X, X_expected))

    def test_nullspace(self):
        A = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]])

        # Find nullspace
        x = nullspace(A)

        # Check nullspace
        y = np.dot(A, x)
        res = np.abs(y).max()
        self.assertTrue(abs(res) < 0.0000001)
