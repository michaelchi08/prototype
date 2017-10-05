import unittest
from os.path import join

import numpy as np
import matplotlib.pylab as plt

import prototype.tests as test
from prototype.vision.camera_models import PinholeCameraModel
from prototype.vision.common import camera_intrinsics
from prototype.vision.common import rand3dfeatures


class PinholeCameraModelTest(unittest.TestCase):
    def test_constructor(self):
        K = camera_intrinsics(554.25, 554.25, 320.0, 320.0)
        camera = PinholeCameraModel(640, 640, 10, K)

        self.assertEqual(camera.image_width, 640)
        self.assertEqual(camera.image_height, 640)
        self.assertEqual(camera.hz, 10)
        self.assertEqual(camera.dt, 0)
        self.assertEqual(camera.frame, 0)

    def test_update(self):
        K = camera_intrinsics(554.25, 554.25, 320.0, 320.0)
        camera = PinholeCameraModel(640, 640, 10, K)
        retval = camera.update(0.11)

        self.assertTrue(retval)
        self.assertEqual(camera.dt, 0.0)
        self.assertEqual(camera.frame, 1)

    def test_check_landmarks(self):
        # setup camera
        K = camera_intrinsics(554.25, 554.25, 320.0, 320.0)
        camera = PinholeCameraModel(640, 640, 10, K)

        # setup random 3d landmarks
        nb_landmarks = 100
        feature_bounds = {
            "x": {"min": -1.0, "max": 10.0},
            "y": {"min": -1.0, "max": 1.0},
            "z": {"min": -1.0, "max": 1.0}
        }
        landmarks = rand3dfeatures(nb_landmarks, feature_bounds)

        # test
        rpy = [0.0, 0.0, 0.0]
        t = [0.0, 0.0, 0.0]
        observed = camera.check_landmarks(0.11, landmarks, rpy, t)

        self.assertTrue(len(observed) > 0)

    def test_example(self):
        # load points
        points_file = join(test.TEST_DATA_PATH, "house/house.p3d")
        points = np.loadtxt(points_file).T
        points = np.vstack((points, np.ones(points.shape[1])))

        # setup camera
        K = np.eye(3)
        R = np.eye(3)
        t = np.array([0, 0, 0])
        camera = PinholeCameraModel(320, 240, 60, K)
        x = camera.project(points, R, t)

        # plot projection
        debug = False
        if debug:
            plt.figure()
            plt.plot(x[0], x[1], 'k. ')
            plt.show()
