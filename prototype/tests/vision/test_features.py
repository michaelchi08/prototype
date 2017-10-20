import sys
import time
import unittest
from os.path import join

import cv2
import numpy as np
import matplotlib.pylab as plt

import prototype.tests as test
from prototype.data.kitti import VOSequence
from prototype.vision.features import Keypoint
from prototype.vision.features import Keyframe
from prototype.vision.features import FASTDetector
from prototype.vision.features import ORBDetector
from prototype.vision.features import LKFeatureTracker
from prototype.vision.features import FeatureTracker

# GLOBAL VARIABLES
VO_DATA_PATH = "/data/dataset"


class KeypointTest(unittest.TestCase):
    def test_init(self):
        kp = Keypoint([1, 2], 31)
        self.assertEqual(kp.pt[0], 1)
        self.assertEqual(kp.pt[1], 2)
        self.assertEqual(kp.size, 31)


class KeyframeTest(unittest.TestCase):
    def test_init(self):
        kf = Keyframe(np.zeros((100, 100)), np.ones((2, 100)))
        self.assertTrue(np.array_equiv(kf.image, np.zeros((100, 100))))
        self.assertTrue(np.array_equiv(kf.features, np.ones((2, 100))))


class FASTDetectorTest(unittest.TestCase):
    def test_detect(self):
        detector = FASTDetector()
        img = cv2.imread(join(test.TEST_DATA_PATH, "empire/empire.jpg"))

        debug = False
        keypoints = detector.detect(img, debug)
        if debug:
            cv2.waitKey(0)

        self.assertTrue(len(keypoints) >= 3800)


class ORBDetectorTest(unittest.TestCase):
    def test_detect(self):
        detector = ORBDetector()
        img = cv2.imread(join(test.TEST_DATA_PATH, "empire/empire.jpg"))

        debug = False
        features = detector.detect(img, debug)
        if debug:
            cv2.waitKey(0)

        self.assertTrue(len(features) >= 100)


class LKFeatureTrackerTest(unittest.TestCase):
    def setUp(self):
        detector = FASTDetector(threshold=150)
        self.tracker = LKFeatureTracker(detector)
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
            plt.clf()


class FeatureTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = FeatureTracker()
        self.data = VOSequence(VO_DATA_PATH, "00")

    def test_initialize(self):
        img = cv2.imread(self.data.image_0_files[0])
        self.tracker.initialize(img)

        self.assertTrue(self.tracker.img_ref is not None)
        self.assertTrue(self.tracker.fea_ref is not None)
        self.assertTrue(len(self.tracker.tracks_alive) > 0)

    def test_detect(self):
        img = cv2.imread(self.data.image_0_files[0])
        features = self.tracker.detect(img)
        self.assertTrue(len(features) > 10)

    def test_match(self):
        img0 = cv2.imread(self.data.image_0_files[0])
        img1 = cv2.imread(self.data.image_0_files[1])

        # Obtain features in both frames
        self.tracker.initialize(img0)
        f0 = self.tracker.fea_ref
        f1 = self.tracker.detect(img1)

        # Perform matching
        matches = self.tracker.match(f0, f1)

        self.assertTrue(len(matches) < len(self.tracker.fea_ref))

    def test_update_feature_tracks(self):
        img0 = cv2.imread(self.data.image_0_files[0])
        img1 = cv2.imread(self.data.image_0_files[1])
        img2 = cv2.imread(self.data.image_0_files[2])
        img3 = cv2.imread(self.data.image_0_files[3])

        # Obtain features from images
        self.tracker.initialize(img0)
        f1 = self.tracker.detect(img1)

        # First match
        matches = self.tracker.match(self.tracker.fea_ref, f1)
        self.tracker.update_feature_tracks(matches, self.tracker.fea_ref, f1)

        self.assertTrue(len(self.tracker.tracks_alive) > 0)
        self.assertEqual(len(self.tracker.tracks_alive), len(matches))
        self.assertEqual(len(self.tracker.tracks_dead), 0)

        # Second match
        f2 = self.tracker.detect(img2)
        matches = self.tracker.match(self.tracker.fea_ref, f2)
        self.tracker.update_feature_tracks(matches, self.tracker.fea_ref, f2)

        self.assertTrue(len(self.tracker.tracks_alive) > 0)
        self.assertEqual(len(self.tracker.tracks_alive), len(matches))
        self.assertTrue(len(self.tracker.tracks_dead) > 0)

        # Third match
        f3 = self.tracker.detect(img3)
        matches = self.tracker.match(self.tracker.fea_ref, f3)
        self.tracker.update_feature_tracks(matches, self.tracker.fea_ref, f3)

        self.assertTrue(len(self.tracker.tracks_alive) > 0)
        self.assertEqual(len(self.tracker.tracks_alive), len(matches))
        self.assertTrue(len(self.tracker.tracks_dead) > 0)

        # Plot feature tracks
        debug = False
        if debug:
            for track_id in self.tracker.tracks_alive:
                track_x = []
                track_y = []
                for feature in self.tracker.tracks_buffer[track_id].track:
                    track_x.append(feature.pt[0])
                    track_y.append(feature.pt[1])
                plt.plot(track_x, track_y)
            plt.show()

    def test_draw_matches(self):
        img0 = cv2.imread(self.data.image_0_files[0])
        img1 = cv2.imread(self.data.image_0_files[1])

        # Obtain features
        self.tracker.initialize(img0)
        f0 = self.tracker.fea_ref
        f1 = self.tracker.detect(img1)

        # Perform matching
        matched = self.tracker.match(f0, f1)

        # Draw matches
        img = self.tracker.draw_matches(img0, img1,
                                        f0, f1,
                                        matched)

        # Show matches
        debug = False
        if debug:
            cv2.imshow("Matches", img)
            cv2.waitKey(0)

    def plot_storage(self, storage):
        plt.figure()
        plt.plot(range(len(storage)), storage)
        plt.title("Num of tracks over time")
        plt.xlabel("Frame No.")
        plt.ylabel("Num of Tracks")

    def plot_tracked(self, tracked):
        plt.figure()
        plt.plot(range(len(tracked)), tracked)
        plt.title("Matches per Frame")
        plt.xlabel("Frame No.")
        plt.ylabel("Num of Tracks")

    def plot_tracks(self, fig, ax):
        ax.cla()
        ax.set_xlim([0, 1000])
        ax.set_ylim([0, 300])
        for track_id in self.tracker.tracks_alive:
            track_x = []
            track_y = []

            if track_id is not None:
                for feature in self.tracker.tracks_buffer[track_id].track:
                    track_x.append(feature.pt[0])
                    track_y.append(feature.pt[1])
                ax.plot(track_x, track_y)
                fig.canvas.draw()

    def test_update(self):
        debug = False

        # Stats
        tracked = []
        storage = []

        # Plot tracks
        # plt.ion()
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.set_title("Tracks")
        # ax.set_xlabel("pixel")
        # ax.set_ylabel("pixel")

        # Loop through images
        index = 0
        nb_images = len(self.data.image_0_files[:200])
        time_start = time.time()

        while index <= nb_images:
            # Index out of bounds guard
            index = 0 if index < 0 else index

            # Feature tracker update
            img = cv2.imread(self.data.image_0_files[index])
            self.tracker.update(img, debug)
            # self.plot_tracks(fig, ax)

            # Record tracker stats
            tracked.append(len(self.tracker.tracks_alive))
            storage.append(len(self.tracker.tracks_buffer))

            # Display image
            if debug:
                # cv2.imshow("VO Sequence " + self.data.sequence, img)
                key = cv2.waitKey(1)
                if key == ord('q'):  # Quit
                    index = nb_images + 1
                elif key == ord('p'):  # Previous image
                    index -= 1
                else:
                    index += 1
            else:
                index += 1

        time_elapsed = time.time() - time_start  # NOQA
        # print("fps: ", float(index) / time_elapsed)

        # self.plot_tracked(tracked)
        # self.plot_storage(storage)
        # plt.show()
