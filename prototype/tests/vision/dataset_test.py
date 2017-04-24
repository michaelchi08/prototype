#!/usr/bin/env python3
import os
import shutil
import unittest

from prototype.vision.dataset import SimCamera
from prototype.vision.dataset import DatasetGenerator
from prototype.vision.common import camera_intrinsics
from prototype.vision.common import random_3d_features


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
        # setup camera
        K = camera_intrinsics(554.25, 554.25, 320.0, 320.0)
        camera = SimCamera(640, 640, 10, K)

        # setup random 3d features
        nb_features = 100
        feature_bounds = {
            "x": {"min": -1.0, "max": 10.0},
            "y": {"min": -1.0, "max": 1.0},
            "z": {"min": -1.0, "max": 1.0}
        }
        features = random_3d_features(nb_features, feature_bounds)

        # test
        rpy = [0.0, 0.0, 0.0]
        t = [0.0, 0.0, 0.0]
        observed = camera.check_features(0.11, features, rpy, t)

        self.assertTrue(len(observed) > 0)


class DatasetGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.save_dir = "/tmp/dataset_test"
        if os.path.isdir(self.save_dir):
            shutil.rmtree(self.save_dir)
        self.dataset = DatasetGenerator()

    def test_generate_test_data(self):
        self.dataset.generate_test_data(self.save_dir)
