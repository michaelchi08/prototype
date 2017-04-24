#!/usr/bin/env python3
import unittest

import numpy as np

from prototype.vision.dataset import SimCamera
from prototype.vision.utils import camera_intrinsics


class SimCameraTest(unittest.TestCase):
    def test_constructor(self):
        K = camera_intrinsics(554.25, 554.25, 320.0, 320.0)
        camera = SimCamera(640, 640, 10, K)

        self.assertEqual(camera.image_width, 640)
        self.assertEqual(camera.image_height, 640)
        self.assertEqual(camera.hz, 10)
        self.assertEqual(camera.dt, 0)
        self.assertEqual(camera.frame, 0)

    def test_update(self):
        K = camera_intrinsics(554.25, 554.25, 320.0, 320.0)
        camera = SimCamera(640, 640, 10, K)
        retval = camera.update(0.11)

        self.assertTrue(retval)
        self.assertEqual(camera.dt, 0.0)
        self.assertEqual(camera.frame, 1)

    def test_check_features(self):
        K = np.array([[], [], []])
        camera = SimCamera(640, 640, 10, K)

        camera.check_features(0.11)
