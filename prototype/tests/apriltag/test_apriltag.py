import unittest

import cv2

from prototype.vision.camera import Camera
from prototype.apriltag.apriltag import AprilTagDetector


class ApriltagTest(unittest.TestCase):
    def test_detect(self):
        # Camera
        camera = Camera()
        detector = AprilTagDetector()

        while cv2.waitKey(1) != 113:
            img = camera.update()
            results = detector.detect(img)

            # Draw corners
            for tag in results:
                for corner in tag.p:
                    pt = (int(corner[0]), int(corner[1]))
                    img = cv2.circle(img, pt, 10, (0, 255, 0), -1)

            # Show image
            cv2.imshow("Image", img)
