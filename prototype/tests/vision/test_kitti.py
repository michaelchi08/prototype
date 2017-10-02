import unittest

from prototype.vision.kitti import parse_data_dir
from prototype.vision.kitti import benchmark_mono_vo


class KittiTest(unittest.TestCase):
    def test_parse_data_dir(self):
        path = "/data/dataset/sequences/00/image_0"
        img_files, nb_imgs = parse_data_dir(path)
        self.assertTrue(len(img_files) > 1)
        self.assertTrue(nb_imgs > 1)

    def test_benchmark_kitti_mono_vo(self):
        data_path = "/data/dataset/sequences/"
        sequence = "00"
        # benchmark_mono_vo(data_path, sequence, None, visualize=True)
