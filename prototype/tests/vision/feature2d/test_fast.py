import unittest
from os.path import join

import cv2

import prototype.tests as test
from prototype.vision.feature2d.fast import FAST
from prototype.vision.feature2d.common import draw_keypoints


class FASTTest(unittest.TestCase):
    def test_detect(self):
        fast = FAST()
        img = cv2.imread(join(test.TEST_DATA_PATH, "empire/empire.jpg"))
        kps = fast.detect_keypoints(img)

        # Debug
        # debug = True
        debug = False
        if debug:
            img = draw_keypoints(img, kps)
            cv2.imshow("image", img)
            cv2.waitKey(0)

        self.assertTrue(len(kps) >= 3800)
