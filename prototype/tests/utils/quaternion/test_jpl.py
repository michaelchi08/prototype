import unittest

import numpy as np

from prototype.utils.utils import deg2rad
from prototype.utils.utils import rotnormalize
from prototype.utils.quaternion.jpl import quatnorm
from prototype.utils.quaternion.jpl import quatnormalize
from prototype.utils.quaternion.jpl import quatconj
from prototype.utils.quaternion.jpl import quatmul
from prototype.utils.quaternion.jpl import quat2rot
from prototype.utils.quaternion.jpl import Omega
from prototype.utils.quaternion.jpl import quatlcomp
from prototype.utils.quaternion.jpl import quatrcomp
from prototype.utils.quaternion.jpl import euler2quat


class JPLQuaternionTest(unittest.TestCase):
    def test_quatnorm_and_quatnormalize(self):
        q = np.array([1.0, 2.0, 3.0, 1.0])
        q = quatnormalize(q)
        q_norm = quatnorm(q)
        self.assertEqual(q_norm, 1.0)

    def test_quatconj(self):
        q = np.array([1.0, 2.0, 3.0, 4.0])
        q_conj = quatconj(q)

        self.assertEqual(q_conj[0], -q[0])
        self.assertEqual(q_conj[1], -q[1])
        self.assertEqual(q_conj[2], -q[2])
        self.assertEqual(q_conj[3], q[3])

    def test_quatmul(self):
        p = np.array([1.0, 2.0, 3.0, 1.0])
        q = np.array([3.0, 2.0, 1.0, 1.0])
        p = quatnormalize(p)
        q = quatnormalize(q)

        p_conj = quatconj(p)
        q_conj = quatconj(q)

        # p^-1 * q^-1
        pq_conj = quatmul(p_conj, q_conj)

        # (q * p)^-1
        qp_conj = quatconj(quatmul(q, p))

        # p^-1 * q^-1 = (q * p)^-1
        self.assertTrue(np.array_equal(pq_conj, qp_conj))

    def test_quat2rot(self):
        q = np.array([0.0, 0.0, 0.0, 1.0])
        R = quat2rot(q)
        self.assertTrue(np.array_equal(R, np.eye(3)))

    def test_Omega(self):
        X = Omega(np.array([1.0, 2.0, 3.0]))
        expected = np.array([[-0.0,  3.0, -2.0, 1.0],
                             [-3.0, -0.0, 1.0, 2.0],
                             [2.0, -1.0, -0.0, 3.0],
                             [-1.0, -2.0, -3.0, 0.0]])

        self.assertEqual(X.shape, (4, 4))
        self.assertTrue(np.array_equal(X, expected))

    def test_quatlcomp(self):
        X = quatlcomp(np.array([1.0, 2.0, 3.0, 1.0]))
        expected = np.array([[1.0, 3.0, -2.0, 1.0],
                             [-3.0, 1.0, 1.0, 2.0],
                             [2.0, -1.0, 1.0, 3.0],
                             [-1.0, -2.0, -3.0, 1.0]])
        self.assertEqual(X.shape, (4, 4))
        self.assertTrue(np.array_equal(X, expected))

    def test_quatrcomp(self):
        X = quatrcomp(np.array([1.0, 2.0, 3.0, 1.0]))
        expected = np.array([[1.0, -3.0, 2.0, 1.0],
                             [3.0, 1.0, -1.0, 2.0],
                             [-2.0, 1.0, 1.0, 3.0],
                             [-1.0, -2.0, -3.0, 1.0]])
        self.assertEqual(X.shape, (4, 4))
        self.assertTrue(np.array_equal(X, expected))

    # def test_sandbox(self):
    #     roll = 10.0
    #     pitch = 20.0
    #     yaw = 30.0
    #     euler = np.array([deg2rad(roll), deg2rad(pitch), deg2rad(yaw)])
    #     q = euler2quat(euler)
    #
    #     R1 = quat2rot(q)
    #
    #     R2 = np.dot(quatrcomp(q).T, quatlcomp(q))
    #     R2 = rotnormalize(R2[0:3, 0:3])
    #
    #     print()
    #     print("R1:\n", R1)
    #     print(np.dot(R1, x))
    #     print("R2:\n", R2)
    #     print(np.dot(R2, x))
