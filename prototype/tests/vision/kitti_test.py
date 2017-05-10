#!/usr/bin/env python3
import unittest

from prototype.vision.kitti import parse_data_dir
from prototype.vision.kitti import benchmark_kitti_mono_vo


class KittiTest(unittest.TestCase):
    def test_parse_data_dir(self):
        path = "/data/kitti_odometry/dataset/sequences/00/image_0"
        img_files, nb_imgs = parse_data_dir(path)
        self.assertTrue(len(img_files) > 1)
        self.assertTrue(nb_imgs > 1)

    def test_benchmark_kitti_mono_vo(self):
        data_path = "/data/kitti_odometry/dataset/sequences/"
        sequences = "00"
        benchmark_kitti_mono_vo(data_path, sequences, None, visualize=True)
