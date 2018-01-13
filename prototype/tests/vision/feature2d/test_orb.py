import unittest
from os.path import join

import cv2

import prototype.tests as test
from prototype.vision.feature2d.orb import ORB
from prototype.vision.feature2d.common import draw_keypoints
from prototype.vision.feature2d.common import draw_features


class ORBTest(unittest.TestCase):
    def setUp(self):
        self.orb = ORB()
        self.img = cv2.imread(join(test.TEST_DATA_PATH, "empire/empire.jpg"))

    def test_detetct_keypoints(self):
        kps = self.orb.detect_keypoints(self.img)

        # Debug
        # debug = True
        debug = False
        if debug:
            img = draw_keypoints(self.img, kps)
            cv2.imshow("image", img)
            cv2.waitKey(0)

    def test_detect_features(self):
        features = self.orb.detect_features(self.img)

        # Debug
        # debug = True
        debug = False
        if debug:
            self.img = draw_features(self.img, features)
            cv2.imshow("image", self.img)
            cv2.waitKey(0)

        self.assertTrue(len(features) >= 100)

    def test_extract_descriptors(self):
        kps = self.orb.detect_keypoints(self.img)
        kps, des = self.orb.extract_descriptors(self.img, kps)

        self.assertTrue(len(des) > 0)
        self.assertTrue(len(kps), len(des))
