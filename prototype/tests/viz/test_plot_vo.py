import os
import shutil
import unittest

from prototype.vision.dataset import DatasetGenerator
from prototype.viz.plot_vo import load_feature_data
from prototype.viz.plot_vo import load_observed_data
from prototype.viz.plot_vo import plot_3d

import numpy as np
from numpy import dot
from prototype.utils.euler import euler2rot
from prototype.utils.quaternion.hamiltonian import euler2quat
from prototype.vision.common import projection_matrix

DATASET_PATH = "/tmp/dataset_test"


class VOPlotTest(unittest.TestCase):
    def setUp(self):
        self.save_dir = "/tmp/dataset_test"
        if os.path.isdir(self.save_dir):
            shutil.rmtree(self.save_dir)
        self.dataset = DatasetGenerator()

    def tearDown(self):
        if os.path.isdir(self.save_dir):
            shutil.rmtree(self.save_dir)

    def test_plot_3d(self):
        self.dataset.generate_test_data(self.save_dir)
        feature_data = load_feature_data(DATASET_PATH)
        observed_data = load_observed_data(DATASET_PATH)

        self.assertEqual(feature_data.shape, (3, 1000))
        self.assertTrue(observed_data is not None)
        # plot_3d(feature_data, observed_data)

    def test_sandbox(self):
        self.dataset.generate_test_data(self.save_dir)

        feature = np.array([31.0723, 293.864])
        feature = np.array([-1.11603, -0.0716818, 1.41978, 1.0])
        t = np.array([-0.0109597, 0.0, 0.109232])
        rpy = np.array([0.0, -0.22, 0.0])

        R = euler2rot(rpy, 123)
        P = projection_matrix(self.dataset.camera.K, R, dot(-R, t))
        x = dot(P, feature)
        x[0] = x[0] / x[2]
        x[1] = x[1] / x[2]
        x[2] = x[2] / x[2]

        print("quaternion: ", euler2quat(rpy, 321))
        print("R: ", R)
        print("K: ", self.dataset.camera.K)
        print("predicted: ", x[0:2])
        print("actual: ", feature)
