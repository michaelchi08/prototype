import os
import shutil
import unittest

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

    def test_simulate_test_data(self):
        self.dataset.simulate_test_data()

    def test_generate_test_data(self):
        self.dataset.generate_test_data(self.save_dir)
