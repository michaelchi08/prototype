import unittest

import numpy as np


from prototype.utils.quaternion.hamiltonian import quatnorm
from prototype.utils.quaternion.hamiltonian import quatnormalize
from prototype.utils.quaternion.hamiltonian import quatconj
from prototype.utils.quaternion.hamiltonian import quatmul
from prototype.utils.quaternion.hamiltonian import quat2euler
from prototype.utils.quaternion.hamiltonian import quat2rot


class HamiltonianQuaternionTest(unittest.TestCase):
    def test_quatnorm_and_quatnormalize(self):
        q = np.array([1.0, 2.0, 3.0, 1.0])
        q = quatnormalize(q)
        q_norm = quatnorm(q)
        self.assertEqual(q_norm, 1.0)

    def test_quatconj(self):
        q = np.array([1.0, 2.0, 3.0, 4.0])
        q_conj = quatconj(q)

        self.assertEqual(q_conj[0], q[0])
        self.assertEqual(q_conj[1], -q[1])
        self.assertEqual(q_conj[2], -q[2])
        self.assertEqual(q_conj[3], -q[3])

    # def test_quatmul(self):
    #     p = np.array([1.0, 2.0, 3.0, 1.0])
    #     q = np.array([3.0, 2.0, 1.0, 1.0])
    #     p = quatnormalize(p)
    #     q = quatnormalize(q)
    #
    #     p_conj = quatconj(p)
    #     q_conj = quatconj(q)
    #
    #     # p^-1 * q^-1
    #     pq_conj = quatmul(p_conj, q_conj)
    #
    #     # (q * p)^-1
    #     qp_conj = quatconj(quatmul(p, q))
    #
    #     # p^-1 * q^-1 = (q * p)^-1
    #     self.assertTrue(np.array_equal(pq_conj, qp_conj))

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
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = quat2rot(q)
        self.assertTrue(np.array_equal(R, np.eye(3)))
