import unittest

import numpy as np

from prototype.estimation.msckf import MSCKF


class MSCKFTest(unittest.TestCase):
    def test_imu_state_update(self):
        msckf = MSCKF()
        # q_I_W = [0.0, 0.0, 0.0, 0.0]
        # w_W = np.array([[0.0], [0.0], [0.0]])
        dt = 0.0

        msckf.imu_state_update(dt)

    def test_prediction_update(self):
        pass

    def test_measurement_update(self):
        pass
