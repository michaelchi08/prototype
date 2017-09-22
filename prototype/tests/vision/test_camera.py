import unittest

import cv2

from prototype.vision.camera import Camera


class CameraTest(unittest.TestCase):
    def test_camera(self):
        camera = Camera()
        camera.loop()
        pass
