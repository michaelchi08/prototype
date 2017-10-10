import unittest

import numpy as np

from prototype.estimation.msckf import MSCKF


class MSCKFTest(unittest.TestCase):
    def setUp(self):
        self.msckf = MSCKF(n_g=0.001 * np.ones(3).reshape((3, 1)),
                           n_a=0.001 * np.ones(3).reshape((3, 1)),
                           n_wg=0.001 * np.ones(3).reshape((3, 1)),
                           n_wa=0.001 * np.ones(3).reshape((3, 1)))
        pass

    def test_F(self):
        q_hat = np.array([1.0, 0.0, 0.0, 0.0])
        a_hat = np.array([1.0, 2.0, 3.0])
        w_G = np.array([0.0, 0.0, 0.0])

        F = self.msckf.F(q_hat, a_hat, w_G)
        print(F)

    def test_prediction_update(self):
        # q_I_W = [0.0, 0.0, 0.0, 0.0]
        # w_W = np.array([[0.0], [0.0], [0.0]])
        # dt = 0.0
        # self.msckf.prediction_update(dt)
        pass

    def test_measurement_update(self):
        pass
