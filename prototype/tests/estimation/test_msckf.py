import unittest

import sympy
import numpy as np

from prototype.utils.euler import euler2quat
from prototype.utils.quaternion import quat2rot
from prototype.estimation.msckf import skew
from prototype.estimation.msckf import Omega
from prototype.estimation.msckf import C
from prototype.estimation.msckf import derive_ba_inverse_depth_jacobian
from prototype.estimation.msckf import derive_H_f_j
from prototype.estimation.msckf import sandbox
from prototype.estimation.msckf import CameraState
from prototype.estimation.msckf import MSCKF
from prototype.vision.common import focal_length
from prototype.vision.common import camera_intrinsics
from prototype.vision.camera_model import PinholeCameraModel
from prototype.vision.features import Keypoint
from prototype.vision.features import FeatureTrack


class CameraStateTest(unittest.TestCase):
    def test_init(self):
        p_G_C = np.array([1.0, 2.0, 3.0])
        q_C_G = np.array([1.0, 0.0, 0.0, 0.0])
        cam = CameraState(p_G_C, q_C_G)

        self.assertTrue(np.array_equal(p_G_C, cam.p_G_C))
        self.assertTrue(np.array_equal(q_C_G, cam.q_C_G))


class MSCKFTest(unittest.TestCase):
    def setUp(self):
        self.msckf = MSCKF(n_g=0.001 * np.ones(3).reshape((3, 1)),
                           n_a=0.001 * np.ones(3).reshape((3, 1)),
                           n_wg=0.001 * np.ones(3).reshape((3, 1)),
                           n_wa=0.001 * np.ones(3).reshape((3, 1)))
        pass

    # def test_derive_ba_inverse_depth_jacobian(self):
    #     J = derive_ba_inverse_depth_jacobian()
    #
    #     debug = False
    #     if debug:
    #         sympy.pprint(J[0])
    #         sympy.pprint(J[1])
    #         sympy.pprint(J[2])

    def test_derive_H_f_j(self):
        derive_H_f_j()

    def test_skew(self):
        X = skew(np.array([1.0, 2.0, 3.0]))
        X_expected = np.array([[0.0, -3.0, 2.0],
                               [3.0, 0.0, -1.0],
                               [-2.0, 1.0, 0.0]])

        self.assertTrue(np.array_equal(X, X_expected))

    def test_Omega(self):
        X = Omega(np.array([1.0, 2.0, 3.0]))
        self.assertEqual(X.shape, (4, 4))

    def test_sandbox(self):
        sandbox()

    def test_F(self):
        w_hat = np.array([0.0, 0.0, 0.0])
        q_hat = np.array([1.0, 0.0, 0.0, 0.0])
        a_hat = np.array([1.0, 2.0, 3.0])
        w_G = np.array([0.0, 0.0, 0.0])

        F = self.msckf.F(w_hat, q_hat, a_hat, w_G)

        # -- First row --
        self.assertTrue(np.array_equal(F[0:3, 0:3], -skew(w_hat)))
        self.assertTrue(np.array_equal(F[0:3, 3:6], np.ones((3, 3))))
        # -- Third Row --
        self.assertTrue(np.array_equal(F[6:9, 0:3], -C(q_hat).T * skew(a_hat)))
        self.assertTrue(np.array_equal(F[6:9, 6:9], -2.0 * skew(w_G)))
        self.assertTrue(np.array_equal(F[6:9, 9:12], -C(q_hat).T))
        self.assertTrue(np.array_equal(F[6:9, 12:15], -skew(w_G)**2))
        # -- Fifth Row --
        self.assertTrue(np.array_equal(F[12:15, 6:9], np.ones((3, 3))))

    def test_G(self):
        q_hat = np.array([1.0, 0.0, 0.0, 0.0])

        G = self.msckf.G(q_hat)

        # -- First row --
        self.assertTrue(np.array_equal(G[0:3, 0:3], np.ones((3, 3))))
        # -- Second row --
        self.assertTrue(np.array_equal(G[3:6, 3:6], np.ones((3, 3))))
        # -- Third row --
        self.assertTrue(np.array_equal(G[6:9, 6:9], -C(q_hat).T))
        # -- Fourth row --
        self.assertTrue(np.array_equal(G[9:12, 9:12], np.ones((3, 3))))

    # def test_prediction_update(self):
    #     q_I_W = [0.0, 0.0, 0.0, 0.0]
    #     w_W = np.array([[0.0], [0.0], [0.0]])
    #     dt = 0.0
    #     self.msckf.prediction_update(dt)
    #     pass

    def test_estimate_feature(self):
        # Pinhole Camera model
        image_width = 640
        image_height = 480
        fov = 60
        fx, fy = focal_length(image_width, image_height, fov)
        cx, cy = (image_width / 2.0, image_height / 2.0)
        K = camera_intrinsics(fx, fy, cx, cy)
        cam_model = PinholeCameraModel(image_width, image_height, K)

        # Camera states
        cam_states = []
        # -- Camera state 0
        p_G_C0 = np.array([0.0, 0.0, 0.0])
        q_C0_G = np.array([1.0, 0.0, 0.0, 0.0])
        cam_states.append(CameraState(p_G_C0, q_C0_G))
        # -- Camera state 1
        p_G_C1 = np.array([1.0, 0.0, 0.0])
        q_C1_G = euler2quat([0.1, 0.0, 0.0], 321)
        cam_states.append(CameraState(p_G_C1, q_C1_G))

        # Features
        landmark = np.array([1.0, 2.0, 10.0, 1.0])
        kp1 = cam_model.project(landmark, quat2rot(q_C0_G), p_G_C0) + np.array([0.1, 0.1, 0.0])
        kp2 = cam_model.project(landmark, quat2rot(q_C1_G), p_G_C1) + np.array([0.1, 0.1, 0.0])
        kp1 = Keypoint(kp1[:2], 21)
        kp2 = Keypoint(kp2[:2], 21)
        track = FeatureTrack(0, 1, kp1, kp2)

        # Estimate feature
        results = self.msckf.estimate_feature(cam_model, cam_states, track)
        k, r, alpha, beta, rho = results

        self.assertTrue(k < 10)
        self.assertTrue(alpha - landmark[0] < 0.1)
        self.assertTrue(beta - landmark[1] < 0.1)
        self.assertTrue((1 / rho) - landmark[2] < 0.1)

    def test_measurement_update(self):
        pass
