import unittest
from prototype.vision.feature2d.keypoint import KeyPoint


class KeyPointTest(unittest.TestCase):
    def test_init(self):
        kp = KeyPoint([1, 2], 31)
        self.assertEqual(kp.pt[0], 1)
        self.assertEqual(kp.pt[1], 2)
        self.assertEqual(kp.size, 31)
