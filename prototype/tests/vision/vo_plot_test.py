#!/usr/bin/env python3
import os
import shutil
import unittest

from prototype.vision.dataset import DatasetGenerator
from prototype.vision.vo_plot import load_features_data
from prototype.vision.vo_plot import load_all_observation_data
from prototype.vision.vo_plot import plot_3d

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
        fea_data = load_features_data(DATASET_PATH)
        obs_data = load_all_observation_data(DATASET_PATH)

        self.assertEqual(fea_data.shape, (3, 100))
        self.assertTrue(obs_data is not None)
        # plot_3d(fea_data, obs_data, True)
