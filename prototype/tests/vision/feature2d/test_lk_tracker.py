import sys
import unittest
from os.path import join

import cv2

import prototype.tests as test
from prototype.vision.feature2d.fast import FAST
from prototype.vision.feature2d.lk_tracker import LKFeatureTracker


class LKFeatureTrackerTest(unittest.TestCase):
    def setUp(self):
        detector = FAST(threshold=150)
        self.tracker = LKFeatureTracker(detector=detector)

        data_path = join(test.TEST_DATA_PATH, "vo")
        self.img = []
        for i in range(10):
            img_filename = "%d.png" % i
            self.img.append(cv2.imread(join(data_path, img_filename)))

    def test_detect(self):
        self.tracker.detect(self.img[0])

        self.assertTrue(len(self.tracker.tracks) > 0)
        self.assertEqual(self.tracker.track_id,
                         len(self.tracker.tracks))
        self.assertEqual(self.tracker.track_id,
                         len(self.tracker.tracks_tracking))

    def test_last_keypoints(self):
        self.tracker.detect(self.img[0])
        keypoints = self.tracker.last_keypoints()

        self.assertEqual(len(self.tracker.tracks_tracking), keypoints.shape[0])
        self.assertEqual(2, keypoints.shape[1])

    def test_track_features(self):
        self.tracker.detect(self.img[0])
        tracks_tracking_before = len(self.tracker.tracks_tracking)

        self.tracker.track_features(self.img[0], self.img[1])
        tracks_tracking_after = len(self.tracker.tracks_tracking)

        self.assertTrue(tracks_tracking_after <= tracks_tracking_before)

    def test_draw_tracks(self):
        debug = False
        self.tracker.detect(self.img[0])
        self.tracker.track_features(self.img[0], self.img[1])
        self.tracker.draw_tracks(self.img[1], debug)

        if debug:
            cv2.waitKey(1000000)

    def test_update(self):
        debug = False
        tracks_tracked = []

        # Loop through images
        index = 0
        while index < len(self.img):
            # Index out of bounds guard
            index = 0 if index < 0 else index

            # Feature tracker update
            self.tracker.update(self.img[index], debug)
            tracks_tracked.append(len(self.tracker.tracks_tracking))

            # Display image
            if debug:
                cv2.imshow("Sequence " + self.data.sequence, self.img[index])
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
            plt.clf()
