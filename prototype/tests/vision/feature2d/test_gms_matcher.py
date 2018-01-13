import os
import time
import unittest

import cv2

import prototype.tests as test
from prototype.vision.camera.camera import Camera
from prototype.vision.feature2d.gms_matcher import GmsMatcher
from prototype.vision.feature2d.gms_matcher import draw_matches


class GMSMatcherTest(unittest.TestCase):
    def setUp(self):
        data_path = os.path.join(test.TEST_DATA_PATH, "vo")
        self.img0 = cv2.imread(os.path.join(data_path, "0.png"))
        self.img1 = cv2.imread(os.path.join(data_path, "1.png"))
        self.img2 = cv2.imread(os.path.join(data_path, "2.png"))

    def test_compute_matches(self):
        orb = cv2.ORB_create(10000)
        orb.setFastThreshold(0)
        gms = GmsMatcher()

        kp0, des0 = orb.detectAndCompute(self.img0, None)
        kp1, des1 = orb.detectAndCompute(self.img1, None)

        kps0, kps1, matches = gms.compute_matches(kp0, kp1,
                                                  des0, des1,
                                                  self.img0.shape)
        matches_img = draw_matches(self.img0, self.img1, kp0, kp1, matches)

        # Show matches
        # debug = True
        debug = False
        if debug:
            cv2.imshow("Matches", matches_img)
            cv2.waitKey(0)

        self.assertTrue(len(matches) > 0)

    @unittest.skip("Requires Hardware")
    def test_compute_matches2(self):
        orb = cv2.ORB_create(1000)
        orb.setFastThreshold(0)

        camera = Camera()
        img0 = camera.update()
        gms = GmsMatcher()
        kp0, des0 = orb.detectAndCompute(img0, None)

        t_start = int(round(time.time() * 1000))

        while True:
            img1 = camera.update()
            kp1, des1 = orb.detectAndCompute(img1, None)
            kps0, kps1, matches = gms.compute_matches(kp0, kp1,
                                                      des0, des1,
                                                      img0.shape)

            t_end = int(round(time.time() * 1000))
            print(1.0 / (t_end - t_start) * 1000)
            t_start = t_end

            # matches_img = draw_matches(img0, img1, kps0, kps1, matches)
            # cv2.imshow("Mathces", matches_img)
            # if cv2.waitKey(1) == 113:
            #     exit(0)

            img0 = img1

        self.assertTrue(len(matches) > 0)
