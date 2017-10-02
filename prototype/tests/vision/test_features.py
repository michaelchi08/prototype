import os
import sys
import unittest

import cv2

from prototype.data.kitti import VOSequence
from prototype.vision.features import FastDetector
from prototype.vision.features import FeatureTracker

# GLOBAL VARIABLES
VO_DATA_PATH = "/data/dataset"


# class FastDetectorTest(unittest.TestCase):
#     def test_detect(self):
#         detector = FastDetector()
#         img = cv2.imread("data/empire/empire.jpg")
#         keypoints = detector.detect(img)
#         self.assertTrue(len(keypoints) >= 3800)


class FeatureTrackerTest(unittest.TestCase):
    def setUp(self):
        detector = FastDetector(threshold=100)
        self.tracker = FeatureTracker(detector)
        self.data = VOSequence(VO_DATA_PATH, "00")

    def test_update(self):
        # Loop through images
        index = 0
        while index <= len(self.data.image_0_files[:10]):
            # Index out of bounds guard
            index = 0 if index < 0 else index

            # Open image 0
            img_path = self.data.image_0_files[index]
            img = cv2.imread(img_path)

            # Feature tracker update
            self.tracker.update(img, True)

            # Display image
            cv2.imshow("VO Sequence " + self.data.sequence, img)
            key = cv2.waitKey(0)
            if key == ord('q'):  # Quit
                sys.exit(1)
            elif key == ord('p'):  # Previous image
                index -= 1
            else:
                index += 1
