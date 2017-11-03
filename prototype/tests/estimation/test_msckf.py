import unittest

import sympy
import numpy as np
from numpy import dot
from numpy import array_equal

from prototype.utils.linalg import skew
from prototype.utils.quaternion.jpl import quat2rot as C

from prototype.estimation.msckf import CameraState
from prototype.estimation.msckf import MSCKF
from prototype.vision.common import focal_length
from prototype.vision.common import camera_intrinsics
from prototype.vision.camera_model import PinholeCameraModel
from prototype.vision.features import Keypoint
from prototype.vision.features import FeatureTrack


class CameraStateTest(unittest.TestCase):
    def test_init(self):
        p_G = np.array([1.0, 2.0, 3.0])
        q_CG = np.array([1.0, 0.0, 0.0, 0.0])
        cam = CameraState(p_G, q_CG)

        self.assertTrue(array_equal(p_G, cam.p_G.ravel()))
        self.assertTrue(array_equal(q_CG, cam.q_CG.ravel()))


class MSCKFTest(unittest.TestCase):
    def setUp(self):
        self.msckf = MSCKF(n_g=0.001 * np.ones(3),
                           n_a=0.001 * np.ones(3),
                           n_wg=0.001 * np.ones(3),
                           n_wa=0.001 * np.ones(3))

    def create_test_case_1(self):
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
        q_C0_G = np.array([0.0, 0.0, 0.0, 1.0])
        cam_states.append(CameraState(p_G_C0, q_C0_G))
        # -- Camera state 1
        p_G_C1 = np.array([1.0, 0.0, 0.0])
        q_C1_G = np.array([0.0, 0.0, 0.0, 1.0])
        cam_states.append(CameraState(p_G_C1, q_C1_G))

        # Features
        landmark = np.array([1.0, 2.0, 10.0, 1.0])
        noise = np.array([0.2, 0.2, 0.0])
        R_C0_G = np.array(C(q_C0_G))
        R_C1_G = np.array(C(q_C1_G))
        kp1 = cam_model.project(landmark, R_C0_G, p_G_C0) + noise
        kp2 = cam_model.project(landmark, R_C1_G, p_G_C1) + noise
        kp1 = Keypoint(kp1[:2], 21)
        kp2 = Keypoint(kp2[:2], 21)
        track = FeatureTrack(0, 1, kp1, kp2)

        return (cam_model, track, cam_states, landmark)

    def test_F(self):
        w_hat = np.array([0.0, 0.0, 0.0])
        q_hat = np.array([1.0, 0.0, 0.0, 0.0])
        a_hat = np.array([1.0, 2.0, 3.0])
        w_G = np.array([0.0, 0.0, 0.0])

        F = self.msckf.F(w_hat, q_hat, a_hat, w_G)

        # -- First row --
        self.assertTrue(array_equal(F[0:3, 0:3], -skew(w_hat)))
        self.assertTrue(array_equal(F[0:3, 3:6], np.ones((3, 3))))
        # -- Third Row --
        self.assertTrue(array_equal(F[6:9, 0:3], dot(-C(q_hat).T, skew(a_hat))))
        self.assertTrue(array_equal(F[6:9, 6:9], -2.0 * skew(w_G)))
        self.assertTrue(array_equal(F[6:9, 9:12], -C(q_hat).T))
        self.assertTrue(array_equal(F[6:9, 12:15], -skew(w_G)**2))
        # -- Fifth Row --
        self.assertTrue(array_equal(F[12:15, 6:9], np.ones((3, 3))))

    def test_G(self):
        q_hat = np.array([1.0, 0.0, 0.0, 0.0]).reshape((4, 1))
        G = self.msckf.G(q_hat)

        # -- First row --
        self.assertTrue(array_equal(G[0:3, 0:3], np.ones((3, 3))))
        # -- Second row --
        self.assertTrue(array_equal(G[3:6, 3:6], np.ones((3, 3))))
        # -- Third row --
        self.assertTrue(array_equal(G[6:9, 6:9], -C(q_hat).T))
        # -- Fourth row --
        self.assertTrue(array_equal(G[9:12, 9:12], np.ones((3, 3))))

    def test_J(self):
        pass

    def test_H(self):
        pass

    def test_prediction_update(self):
        a_m = np.array([[0.0], [0.0], [10.0]])
        w_m = np.array([[0.0], [0.0], [0.0]])
        dt = 0.1

        for i in range(10):
            self.msckf.prediction_update(a_m, w_m, dt)

    def test_estimate_feature(self):
        # Generate test case
        data = self.create_test_case_1()
        (cam_model, track, track_cam_states, landmark) = data

        # Estimate feature
        results = self.msckf.estimate_feature(cam_model,
                                              track,
                                              track_cam_states)
        p_G_f, k, r = results

        # Debug
        debug = False
        if debug:
            print("\nk:", k)
            print("landmark:\n", landmark)
            print("p_G_f:\n", p_G_f)

        # Assert
        self.assertTrue(k < 10)
        self.assertTrue(abs(p_G_f[0, 0] - landmark[0]) < 0.1)
        self.assertTrue(abs(p_G_f[1, 0] - landmark[1]) < 0.1)
        self.assertTrue(abs(p_G_f[2, 0] - landmark[2]) < 0.1)

    def test_calculate_track_residual(self):
        # Generate test case
        data = self.create_test_case_1()
        (cam_model, track, track_cam_states, landmark) = data

        # Estimate feature
        results = self.msckf.estimate_feature(cam_model,
                                              track,
                                              track_cam_states)
        p_G_f, k, r = results

        # Calculate track residual
        debug = False
        residual = self.msckf.calculate_track_residual(cam_model,
                                                       track,
                                                       track_cam_states,
                                                       p_G_f,
                                                       debug)
        if debug:
            print("residual:", residual)

    def test_state_augmentation(self):
        pass

    # def test_measurement_update(self):
    #     pass
