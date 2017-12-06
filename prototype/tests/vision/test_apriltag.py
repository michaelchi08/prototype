import unittest
from os.path import join

import cv2

import prototype.tests as test
from prototype.vision.apriltag import AprilTagDetector


class ApriltagTest(unittest.TestCase):
    def test_detect(self):
        detector = AprilTagDetector()

        img = cv2.imread(join(test.TEST_DATA_PATH, "apriltag", "test.jpg"))
        results = detector.detect(img)
        self.assertTrue(len(results) > 0)

        debug = False
        # debug = True
        if debug:
            # Draw corners
            for tag in results:
                tag.draw_corners(img)
                tag.draw_id(img)

            # Show image
            cv2.imshow("Image", img)
            cv2.waitKey()

    # def test_detect2(self):
    #     from prototype.vision.camera import Camera
    #     detector = AprilTagDetector()
    #     camera = Camera()
    #
    #     # img = cv2.imread(join(test.TEST_DATA_PATH, "apriltag", "test.jpg"))
    #
    #     while cv2.waitKey(1) != 113:
    #         img = camera.update()
    #         results = detector.detect(img)
    #
    #         # Draw corners
    #         for tag in results:
    #             tag.draw_corners(img)
    #
    #         # Show image
    #         cv2.imshow("Image", img)
