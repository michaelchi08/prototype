import os
import unittest

from prototype.data.dataset import download
from prototype.data.dataset import download_kitti_vo_dataset


class DatasetTest(unittest.TestCase):
    def test_download(self):
        url = "http://google.com/index.html"
        output_path = "/tmp"
        download(url, output_path)
        os.remove(os.path.join(output_path, "index.html"))

    # def test_download_kitti_vo_dataset(self):
    #     download_kitti_vo_dataset("/data")
