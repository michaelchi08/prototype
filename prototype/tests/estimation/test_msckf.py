import unittest

import numpy as np

from prototype.estimation.msckf import MSCKF


class MSCKFTest(unittest.TestCase):
    def setUp(self):
        self.msckf = MSCKF()

    def test_F(self):
        q_hat = np.array([1.0, 0.0, 0.0, 0.0])
        a_hat = np.array([1.0, 2.0, 3.0])
        w_G = np.array([0.0, 0.0, 0.0])

        F = self.msckf.F(q_hat, a_hat, w_G)
        print(F)

    # def test_imu_state_update(self):
    #     # q_I_W = [0.0, 0.0, 0.0, 0.0]
    #     # w_W = np.array([[0.0], [0.0], [0.0]])
    #     dt = 0.0
    #     self.msckf.imu_state_update(dt)

    def test_prediction_update(self):
        pass

    def test_measurement_update(self):
        pass
