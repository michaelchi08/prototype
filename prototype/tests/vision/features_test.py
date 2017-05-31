#!/usr/bin/env python3
import unittest

import cv2

from prototype.vision.features import FastDetector


class FastDetectorTest(unittest.TestCase):
    def test_detect(self):
        detector = FastDetector()
        img = cv2.imread("data/empire/empire.jpg")
        keypoints = detector.detect(img)
        self.assertEqual(len(keypoints), 3866)
