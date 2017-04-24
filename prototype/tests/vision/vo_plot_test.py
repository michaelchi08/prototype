#!/usr/bin/env python3
import unittest

from prototype.vision.vo_plot import load_features_data
from prototype.vision.vo_plot import load_all_observation_data
from prototype.vision.vo_plot import plot_3d

DATASET_PATH = "/tmp/dataset_test"


class VOPlotTest(unittest.TestCase):
    def test_plot_3d(self):
        fea_data = load_features_data(DATASET_PATH)
        obs_data = load_all_observation_data(DATASET_PATH)
        plot_3d(fea_data, obs_data)
