import unittest

import numpy as np

from prototype.vision.common import focal_length
from prototype.vision.common import projection_matrix
from prototype.vision.common import factor_projection_matrix
from prototype.vision.common import camera_center
from prototype.vision.common import camera_intrinsics
from prototype.vision.common import rand3dfeatures


class CommonTest(unittest.TestCase):
    def test_focal_length(self):
        fx, fy = focal_length(640, 320, 60)
        self.assertEqual(round(fx, 2), 554.26)
        self.assertEqual(round(fy, 2), 277.13)

    def test_projection_matrix(self):
        # setup
        K = np.array([[554.26, 0, 320],
                      [0, 554.26, 320],
                      [0, 0, 1]])
        R = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        t = np.array([1, 2, 3])

        # test
        P = projection_matrix(K, R, t)

        # assert
        expect = np.array([[554.26, 0, 320, 1514.26],
                           [0, 554.26, 320, 2068.52],
                           [0, 0, 1, 3]])
        self.assertTrue((P == expect).all())

    def test_factor_projection_matrix(self):
        # setup
        P = np.array([[554.26, 0, 320, 1514.26],
                      [0, 554.26, 320, 2068.52],
                      [0, 0, 1, 3]])

        # test
        K, R, t = factor_projection_matrix(P)

        # assert
        K_exp = np.array([[554.26, 0, 320],
                          [0, 554.26, 320],
                          [0, 0, 1]])
        R_exp = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])
        t_exp = np.array([1, 2, 3])
        self.assertTrue((K == K_exp).all())
        self.assertTrue((R == R_exp).all())
        self.assertTrue((t == t_exp).all())

    def test_camera_center(self):
        # setup
        P = np.array([[554.26, 0, 320, 1514.26],
                      [0, 554.26, 320, 2068.52],
                      [0, 0, 1, 3]])

        # test and assert
        C = camera_center(P)
        C_exp = np.array([1, 2, 3])
        self.assertTrue((C == C_exp).all())

    def test_camera_intrinsics(self):
        # setup
        fx = 1.0
        fy = 2.0
        cx = 3.0
        cy = 4.0

        # test and assert
        K = camera_intrinsics(fx, fy, cx, cy)
        K_exp = np.array([[1.0, 0.0, 3.0],
                          [0.0, 2.0, 4.0],
                          [0.0, 0.0, 1.0]])
        self.assertTrue((K == K_exp).all())

    def test_rand3dfeatures(self):
        bounds = {
            "x": {"min": 1.0, "max": 2.0},
            "y": {"min": 3.0, "max": 4.0},
            "z": {"min": 5.0, "max": 6.0}
        }
        features = rand3dfeatures(10, bounds)

        self.assertTrue(features.shape, (3, 10))
        self.assertTrue(np.min(features[0, :]) >= bounds["x"]["min"])
        self.assertTrue(np.max(features[0, :]) <= bounds["x"]["max"])
        self.assertTrue(np.min(features[1, :]) >= bounds["y"]["min"])
        self.assertTrue(np.max(features[1, :]) <= bounds["y"]["max"])
        self.assertTrue(np.min(features[2, :]) >= bounds["z"]["min"])
        self.assertTrue(np.max(features[2, :]) <= bounds["z"]["max"])
