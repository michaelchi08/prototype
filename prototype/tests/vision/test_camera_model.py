import unittest
from os.path import join

import numpy as np
import matplotlib.pylab as plt

import prototype.tests as test
from prototype.vision.common import camera_intrinsics
from prototype.vision.common import rand3dfeatures
from prototype.vision.common import convert2homogeneous
from prototype.vision.camera_model import PinholeCameraModel


class PinholeCameraModelTest(unittest.TestCase):
    def test_constructor(self):
        K = camera_intrinsics(554.25, 554.25, 320.0, 320.0)
        camera = PinholeCameraModel(640, 640, K)
        self.assertEqual(camera.image_width, 640)
        self.assertEqual(camera.image_height, 640)

    def test_P(self):
        # Setup camera model
        K = camera_intrinsics(554.25, 554.25, 320.0, 320.0)
        camera_model = PinholeCameraModel(640, 640, K)

        # Test
        R = np.eye(3)
        t = np.array([0, 0, 0])
        P = camera_model.P(R, t)

        # Assert
        feature = np.array([[0.0], [0.0], [10.0]])
        x = np.dot(P, convert2homogeneous(feature))
        expected = np.array([[320.0], [320.0], [1.0]])

        # Normalize pixel coordinates
        x[0] /= x[2]
        x[1] /= x[2]
        x[2] /= x[2]
        x = np.array(x)

        self.assertTrue(np.array_equal(x, expected))

    def test_project(self):
        # Load points
        points_file = join(test.TEST_DATA_PATH, "house/house.p3d")
        points = np.loadtxt(points_file).T

        # Setup camera
        K = np.eye(3)
        R = np.eye(3)
        t = np.array([0, 0, 0])
        camera = PinholeCameraModel(320, 240, K)
        x = camera.project(points, R, t)

        # Assert
        self.assertEqual(x.shape, (3, points.shape[1]))
        self.assertTrue(np.all(x[2, :] == 1.0))

        # Plot projection
        debug = False
        # debug = True
        if debug:
            plt.figure()
            plt.plot(x[0], x[1], 'k. ')
            plt.show()

    def test_pixel2image(self):
        # Setup camera model
        K = camera_intrinsics(554.25, 554.25, 320.0, 320.0)
        camera_model = PinholeCameraModel(640, 640, K)

        # Test
        pixel = np.array([[320.0], [320.0]])
        imgpt = camera_model.pixel2image(pixel)
        expected = np.array([[0.0], [0.0]])

        # Assert
        self.assertTrue(np.array_equal(imgpt, expected))

    def test_observed_features(self):
        # Setup camera model
        K = camera_intrinsics(554.25, 554.25, 320.0, 320.0)
        camera_model = PinholeCameraModel(640, 640, K)

        # Setup random 3d features
        nb_features = 100
        feature_bounds = {
            "x": {"min": -1.0, "max": 10.0},
            "y": {"min": -1.0, "max": 1.0},
            "z": {"min": -1.0, "max": 1.0}
        }
        features = rand3dfeatures(nb_features, feature_bounds)

        # Test
        rpy = np.array([0.0, 0.0, 0.0])
        t = np.array([0.0, 0.0, 0.0])
        observed = camera_model.observed_features(features, rpy, t)

        # Assert
        self.assertTrue(len(observed) > 0)
