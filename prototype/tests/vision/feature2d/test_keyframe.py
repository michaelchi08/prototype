import unittest

import numpy as np

from prototype.vision.feature2d.keyframe import KeyFrame


class KeyFrameTest(unittest.TestCase):
    def test_init(self):
        kf = KeyFrame(np.zeros((100, 100)), np.ones((2, 100)))
        self.assertTrue(np.array_equiv(kf.image, np.zeros((100, 100))))
        self.assertTrue(np.array_equiv(kf.features, np.ones((2, 100))))
