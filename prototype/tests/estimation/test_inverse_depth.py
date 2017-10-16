import unittest

import numpy as np

from prototype.estimation.inverse_depth import h_C
from prototype.estimation.inverse_depth import feature_init
from prototype.estimation.inverse_depth import camera_motion_model
from prototype.estimation.inverse_depth import linearity_index_inverse_depth
from prototype.estimation.inverse_depth import linearity_index_xyz_parameterization


class InverseDepthTest(unittest.TestCase):
    def test_linearity_index_inverse_depth(self):
        linearity_index_inverse_depth()

    def test_linearity_index_xyz_parameterization(self):
        linearity_index_xyz_parameterization()

    # def test_camera_motion_model(self):
    #     camera_motion_model(X, dt)

    def test_h_C(self):
        y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        r_W_C = np.array([0.0, 0.0, 0.0]).reshape((3, 1))
        q_WC = np.array([1.0, 0.0, 0.0, 0.0]).reshape((4, 1))
        h_C(y, r_W_C, q_WC)

    def test_feature_init(self):
        r_W_C = np.array([0.0, 0.0, 0.0]).reshape((3, 1))
        q_WC = np.array([1.0, 0.0, 0.0, 0.0]).reshape((4, 1))
        pixel = np.array([0.0, 0.0]).reshape((2, 1))
        y = feature_init(r_W_C, q_WC, pixel)

        self.assertEqual(y.shape, (6, 1))
