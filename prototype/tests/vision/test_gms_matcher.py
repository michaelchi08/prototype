import os
import unittest

import cv2
import numpy as np

import prototype.tests as test
from prototype.vision.gms_matcher import GmsMatcher


class GMSMatcherTest(unittest.TestCase):
    def setUp(self):
        data_path = os.path.join(test.TEST_DATA_PATH, "vo")
        self.img0 = cv2.imread(os.path.join(data_path, "0.png"))
        self.img1 = cv2.imread(os.path.join(data_path, "1.png"))
        self.img2 = cv2.imread(os.path.join(data_path, "2.png"))

    def test_compute_matches(self):
        orb = cv2.ORB_create(10000)
        orb.setFastThreshold(0)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        gms = GmsMatcher()

        kp0, des0 = orb.detectAndCompute(self.img0, np.array([]))
        kp1, des1 = orb.detectAndCompute(self.img1, np.array([]))
        matches = matcher.match(des0, des1)

        matches = gms.compute_matches(kp0, kp1, des0, des1, matches, self.img0)

        self.assertTrue(len(matches) > 0)
