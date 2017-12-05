import os
import unittest

import cv2
import numpy as np

import prototype.tests as test
from prototype.vision.gms_matcher import GmsMatcher
from prototype.vision.gms_matcher import draw_matches


class GMSMatcherTest(unittest.TestCase):
    def setUp(self):
        data_path = os.path.join(test.TEST_DATA_PATH, "vo")
        self.img0 = cv2.imread(os.path.join(data_path, "0.png"))
        self.img1 = cv2.imread(os.path.join(data_path, "1.png"))
        self.img2 = cv2.imread(os.path.join(data_path, "2.png"))

    def test_compute_matches(self):
        # cv2.imshow("image", self.img0)
        # cv2.waitKey(0)

        orb = cv2.ORB_create(10000)
        orb.setFastThreshold(0)
        if cv2.__version__.startswith('3'):
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
        gms = GmsMatcher(orb, matcher)

        kp0, des0 = orb.detectAndCompute(self.img0, np.array([]))
        kp1, des1 = orb.detectAndCompute(self.img1, np.array([]))
        # all_matches = self.matcher.match(descriptors_image1, descriptors_image2)

        matches = gms.compute_matches(kp0, kp1, des0, des1, self.img0, self.img1)
        draw_matches(self.img0, self.img1, kp0, kp1, matches)
