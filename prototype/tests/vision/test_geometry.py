import unittest

import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA

from prototype.utils.utils import roty
from prototype.utils.utils import deg2rad
from prototype.vision.common import focal_length
from prototype.vision.common import camera_intrinsics
from prototype.vision.common import rand3dfeatures
from prototype.vision.camera_model import PinholeCameraModel
from prototype.vision.geometry import triangulate
from prototype.vision.geometry import triangulate_point


class GeometryTest(unittest.TestCase):
    def setUp(self):
        # Generate random features
        nb_features = 100
        feature_bounds = {
            "x": {"min": -1.0, "max": 1.0},
            "y": {"min": -1.0, "max": 1.0},
            "z": {"min": 10.0, "max": 20.0}
        }
        features = rand3dfeatures(nb_features, feature_bounds)

        # Pinhole Camera model
        image_width = 640
        image_height = 480
        fov = 60
        fx, fy = focal_length(image_width, image_height, fov)
        cx, cy = (image_width / 2.0, image_height / 2.0)
        K = camera_intrinsics(fx, fy, cx, cy)

        self.cam = PinholeCameraModel(image_width, image_height, K)

        # Rotation and translation of camera 0 and camera 1
        self.R_0 = np.eye(3)
        self.t_0 = np.zeros((3, 1))
        self.R_1 = roty(deg2rad(10.0))
        self.t_1 = np.array([1.0, 0.0, 0.0]).reshape((3, 1))

        # Points as observed by camera 0 and camera 1
        self.features = np.array(features).T
        self.obs0 = self.project_points(features, self.cam, self.R_0, self.t_0)
        self.obs1 = self.project_points(features, self.cam, self.R_1, self.t_1)

    def project_points(self, features, camera, R, t):
        obs = []

        # Make 3D feature homogenenous and project and store pixel measurement
        for f in features:
            f = np.array([f[0], f[1], f[2], 1.0])
            x = camera.project(f, R, t)
            obs.append(x.ravel()[0:2])

        return np.array(obs).T

    def test_triangulate_point(self):
        # Triangulate a single point
        x1 = self.obs0[:, 0]
        x2 = self.obs1[:, 0]
        P1 = self.cam.P(self.R_0, self.t_0)
        P2 = self.cam.P(self.R_1, self.t_1)

        X = triangulate_point(x1, x2, P1, P2)
        X = X[0:3]

        # Assert
        self.assertTrue(np.linalg.norm(X - self.features[:, 0]) < 0.1)

    def test_triangulate(self):
        # Triangulate a set of features
        x1 = self.obs0
        x2 = self.obs1
        P1 = self.cam.P(self.R_0, self.t_0)
        P2 = self.cam.P(self.R_1, self.t_1)

        result = triangulate(x1, x2, P1, P2)

        # Assert
        for i in range(result.shape[1]):
            X = result[:3, i]
            self.assertTrue(np.linalg.norm(X - self.features[:, i]) < 0.1)
