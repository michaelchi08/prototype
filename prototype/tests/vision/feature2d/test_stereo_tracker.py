import unittest
from os.path import join

import cv2
import numpy as np

from prototype.data.kitti import RawSequence
from prototype.vision.feature2d.common import draw_points
from prototype.vision.feature2d.stereo_tracker import StereoTracker


class StereoTrackerTest(unittest.TestCase):
    def test_update2(self):
        # Load KITTI data
        data_path = join("/data", "kitti", "raw")
        data = RawSequence(data_path, "2011_09_26", "0005")

        # Get number of images and image size
        nb_images = len(data.image_00_files)
        img_size = cv2.imread(data.image_00_files[0]).shape
        img_size = (int(img_size[1] / 2), int(img_size[0] / 2))

        # Stereo tracker
        tracker = StereoTracker()

        # Loop through images
        for i in range(nb_images):
            # Load image
            img0 = cv2.imread(data.image_00_files[i])
            img1 = cv2.imread(data.image_01_files[i])

            # Resize image
            img0 = cv2.resize(img0, img_size)
            img1 = cv2.resize(img1, img_size)

            # Detect features
            tracker.update(img0)

            # gyro = data.get_gyroscope(i)
            # print(gyro)

            # Show images
            # img0 = draw_points(img0, corners0)
            # img1 = draw_points(img1, corners1)
            cv2.imshow("Image", np.hstack((img0, img1)))
            if cv2.waitKey(0) == 113:
                break
