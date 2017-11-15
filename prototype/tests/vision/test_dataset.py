import os
import shutil
import unittest

import numpy as np
import matplotlib.pylab as plt

from prototype.vision.dataset import DatasetGenerator


class DatasetGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.save_dir = "/tmp/dataset_test"
        if os.path.isdir(self.save_dir):
            shutil.rmtree(self.save_dir)
        self.dataset = DatasetGenerator()

    def tearDown(self):
        if os.path.isdir(self.save_dir):
            shutil.rmtree(self.save_dir)

    def test_step(self):
        w_B_history = np.zeros((3, 1))
        a_B_history = np.zeros((3, 1))

        for i in range(30):
            (a_B, w_B) = self.dataset.step()
            a_B_history = np.hstack((a_B_history, a_B))
            w_B_history = np.hstack((w_B_history, w_B))

        plt.plot(self.dataset.time, a_B_history[0, :], label="ax")
        plt.plot(self.dataset.time, a_B_history[1, :], label="ay")
        plt.plot(self.dataset.time, a_B_history[2, :], label="az")
        plt.plot(self.dataset.time, w_B_history[0, :], label="wx")
        plt.plot(self.dataset.time, w_B_history[1, :], label="wy")
        plt.plot(self.dataset.time, w_B_history[2, :], label="wz")
        plt.legend(loc=0)
        plt.show()

    def test_simulate_test_data(self):
        self.dataset.simulate_test_data()

    def test_generate_test_data(self):
        self.dataset.generate_test_data(self.save_dir)
