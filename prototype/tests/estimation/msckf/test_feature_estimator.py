import random
import unittest

import numpy as np
from numpy import dot

from prototype.utils.utils import deg2rad
from prototype.utils.quaternion.jpl import quat2rot as C
from prototype.utils.quaternion.jpl import euler2quat
from prototype.vision.common import focal_length
from prototype.vision.common import camera_intrinsics
from prototype.vision.common import rand3dfeatures
from prototype.vision.camera_model import PinholeCameraModel
from prototype.vision.features import Keypoint
from prototype.vision.features import FeatureTrack

from prototype.estimation.msckf.camera_state import CameraState
from prototype.estimation.msckf.feature_estimator import FeatureEstimator


class FeatureEstimatorTest(unittest.TestCase):
    def setUp(self):
        # Pinhole Camera model
        image_width = 640
        image_height = 480
        fov = 60
        fx, fy = focal_length(image_width, image_height, fov)
        cx, cy = (image_width / 2.0, image_height / 2.0)
        K = camera_intrinsics(fx, fy, cx, cy)
        self.cam_model = PinholeCameraModel(image_width, image_height, K)

        # Feature estimator
        self.estimator = FeatureEstimator()

    def test_triangulate(self):
        # Camera states
        # -- Camera state 0
        p_G_C0 = np.array([0.0, 0.0, 0.0])
        rpy_C0G = np.array([deg2rad(0.0), deg2rad(0.0), deg2rad(0.0)])
        q_C0G = euler2quat(rpy_C0G)
        C_C0G = C(q_C0G)
        # -- Camera state 1
        p_G_C1 = np.array([0.0, 1.0, 0.0])
        rpy_C1G = np.array([deg2rad(0.0), deg2rad(0.0), deg2rad(0.0)])
        q_C1G = euler2quat(rpy_C1G)
        C_C1G = C(q_C1G)

        # Features
        landmark = np.array([0.0, 0.0, 10.0])
        kp1 = self.cam_model.project(landmark, C_C0G, p_G_C0)[0:2]
        kp2 = self.cam_model.project(landmark, C_C1G, p_G_C1)[0:2]

        # Calculate rotation and translation of first and last camera states
        # -- Set camera 0 as origin, work out rotation and translation of
        # -- camera 1 relative to to camera 0
        C_C0C1 = dot(C_C0G, C_C1G.T)
        t_C0_C1C0 = dot(C_C0G, (p_G_C1 - p_G_C0))
        # -- Convert from pixel coordinates to image coordinates
        pt1 = self.cam_model.pixel2image(kp1)
        pt2 = self.cam_model.pixel2image(kp2)

        # Triangulate
        p_C1C0_G = self.estimator.triangulate(pt1, pt2, C_C0C1, t_C0_C1C0)

        # Assert
        self.assertTrue(np.allclose(p_C1C0_G.ravel(), landmark))

    def test_estimate_feature(self):
        nb_features = 100
        bounds = {
            "x": {"min": -1.0, "max": 1.0},
            "y": {"min": -1.0, "max": 1.0},
            "z": {"min": 5.0, "max": 10.0}
        }
        features = rand3dfeatures(nb_features, bounds)

        dt = 0.1
        p_G = np.array([0.0, 0.0, 0.0])
        v_G = np.array([0.0, 0.0, 0.1])
        rpy_G = np.array([0.0, 0.0, 0.0])

        # Setup camera states
        track_cam_states = []
        for i in range(10):
            p_G = p_G + v_G * dt
            rpy_G[1] = np.random.normal(0.0, 0.01)
            q_CG = euler2quat(rpy_G)
            track_cam_states.append(CameraState(q_CG, p_G))

        # Feature Track
        start = 0
        end = 10
        feature_idx = random.randint(0, features.shape[1] - 1)
        feature = features[:, feature_idx]

        R_C0G = C(track_cam_states[0].q_CG)
        p_G_C0 = track_cam_states[0].p_G
        R_C1G = C(track_cam_states[1].q_CG)
        p_G_C1 = track_cam_states[1].p_G

        kp1 = self.cam_model.project(feature, R_C0G, p_G_C0)
        kp2 = self.cam_model.project(feature, R_C1G, p_G_C1)
        kp1 = Keypoint(kp1.ravel()[:2], 21)
        kp2 = Keypoint(kp2.ravel()[:2], 21)
        track = FeatureTrack(start, end, kp1, kp2)

        for i in range(2, 10):
            R_CG = C(track_cam_states[i].q_CG)
            p_G_C = track_cam_states[i].p_G
            kp = self.cam_model.project(feature, R_CG, p_G_C)
            kp = Keypoint(kp.ravel()[:2], 21)
            track.update(i, kp)

        # Estimate feature
        p_G_f, k, r = self.estimator.estimate(self.cam_model,
                                              track,
                                              track_cam_states)

        # Debug
        # debug = False
        debug = True
        if debug:
            print("\nk:", k)
            print("feature:\n", feature)
            print("p_G_f:\n", p_G_f)

        # Assert
        # self.assertTrue(k < 10)
        self.assertTrue(abs(p_G_f[0, 0] - feature[0]) < 0.1)
        self.assertTrue(abs(p_G_f[1, 0] - feature[1]) < 0.1)
        self.assertTrue(abs(p_G_f[2, 0] - feature[2]) < 0.1)
