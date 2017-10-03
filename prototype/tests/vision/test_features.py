import sys
import unittest

import cv2

from prototype.data.kitti import VOSequence
from prototype.vision.features import Keypoint
from prototype.vision.features import FastDetector
from prototype.vision.features import FeatureTracker

# GLOBAL VARIABLES
VO_DATA_PATH = "/data/dataset"


class KeypointTest(unittest.TestCase):
    def test_init(self):
        kp = Keypoint([1, 2])
        self.assertEqual(kp.pt[0], 1)
        self.assertEqual(kp.pt[1], 2)


class FastDetectorTest(unittest.TestCase):
    def test_detect(self):
        detector = FastDetector()
        img = cv2.imread("data/empire/empire.jpg")
        keypoints = detector.detect(img)

        self.assertTrue(len(keypoints) >= 3800)


class FeatureTrackerTest(unittest.TestCase):
    def setUp(self):
        detector = FastDetector(threshold=100)
        self.tracker = FeatureTracker(detector)
        self.data = VOSequence(VO_DATA_PATH, "00")

    def test_detect(self):
        img = cv2.imread(self.data.image_0_files[0])
        self.tracker.detect(img)

        self.assertTrue(len(self.tracker.tracks) > 10)
        self.assertEqual(self.tracker.track_id,
                         len(self.tracker.tracks))
        self.assertEqual(self.tracker.track_id,
                         len(self.tracker.tracks_alive))

    def test_last_keypoints(self):
        img = cv2.imread(self.data.image_0_files[0])
        self.tracker.detect(img)
        keypoints = self.tracker.last_keypoints()

        self.assertEqual(len(self.tracker.tracks_alive), keypoints.shape[0])
        self.assertEqual(2, keypoints.shape[1])

    def test_track_features(self):
        img1 = cv2.imread(self.data.image_0_files[0])
        img2 = cv2.imread(self.data.image_0_files[1])

        self.tracker.detect(img1)
        tracks_alive_before = len(self.tracker.tracks_alive)

        self.tracker.track_features(img1, img2)
        tracks_alive_after = len(self.tracker.tracks_alive)

        self.assertTrue(tracks_alive_after < tracks_alive_before)

    def test_draw_tracks(self):
        debug = False
        img1 = cv2.imread(self.data.image_0_files[0])
        img2 = cv2.imread(self.data.image_0_files[1])

        self.tracker.detect(img1)
        self.tracker.track_features(img1, img2)
        self.tracker.draw_tracks(img2, debug)

        if debug:
            cv2.waitKey(1000000)

    def test_update(self):
        debug = False
        tracks_tracked = []

        # Loop through images
        index = 0
        while index <= len(self.data.image_0_files[:100]):
            # Index out of bounds guard
            index = 0 if index < 0 else index

            # Feature tracker update
            img = cv2.imread(self.data.image_0_files[index])
            self.tracker.update(img, debug)
            tracks_tracked.append(len(self.tracker.tracks_alive))

            # Display image
            if debug:
                cv2.imshow("VO Sequence " + self.data.sequence, img)
                key = cv2.waitKey(0)
                if key == ord('q'):  # Quit
                    sys.exit(1)
                elif key == ord('p'):  # Previous image
                    index -= 1
                else:
                    index += 1
            else:
                index += 1

        if debug:
            import matplotlib.pylab as plt
            plt.plot(range(len(tracks_tracked)), tracks_tracked)
            plt.show()
