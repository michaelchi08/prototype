import sys
import time
import unittest
from os.path import join

import cv2
import numpy as np
import matplotlib.pylab as plt

import prototype.tests as test
from prototype.vision.features import KeyPoint
from prototype.vision.features import KeyFrame
from prototype.vision.features import Feature
from prototype.vision.features import FAST
from prototype.vision.features import ORB
from prototype.vision.features import LKFeatureTracker
from prototype.vision.features import FeatureTracker
from prototype.vision.features import draw_keypoints
from prototype.vision.features import draw_features

# GLOBAL VARIABLES
VO_DATA_PATH = "/data/vo"


class KeyPointTest(unittest.TestCase):
    def test_init(self):
        kp = KeyPoint([1, 2], 31)
        self.assertEqual(kp.pt[0], 1)
        self.assertEqual(kp.pt[1], 2)
        self.assertEqual(kp.size, 31)


class KeyFrameTest(unittest.TestCase):
    def test_init(self):
        kf = KeyFrame(np.zeros((100, 100)), np.ones((2, 100)))
        self.assertTrue(np.array_equiv(kf.image, np.zeros((100, 100))))
        self.assertTrue(np.array_equiv(kf.features, np.ones((2, 100))))


class FASTTest(unittest.TestCase):
    def test_detect(self):
        detector = FAST()
        img = cv2.imread(join(test.TEST_DATA_PATH, "empire/empire.jpg"))

        debug = False
        keypoints = detector.detect(img, debug)
        if debug:
            cv2.waitKey(0)

        self.assertTrue(len(keypoints) >= 3800)


class ORBTest(unittest.TestCase):
    def setUp(self):
        self.orb = ORB()
        self.img = cv2.imread(join(test.TEST_DATA_PATH, "empire/empire.jpg"))

    def test_detetct_keypoints(self):
        kps = self.orb.detect_keypoints(self.img)

        # Debug
        # debug = True
        debug = False
        if debug:
            img = draw_keypoints(self.img, kps)
            cv2.imshow("image", img)
            cv2.waitKey(0)

    def test_detect_features(self):
        features = self.orb.detect_features(self.img)

        # Debug
        # debug = True
        debug = False
        if debug:
            self.img = draw_features(self.img, features)
            cv2.imshow("image", self.img)
            cv2.waitKey(0)

        self.assertTrue(len(features) >= 100)

    def test_extract_descriptors(self):
        kps = self.orb.detect_keypoints(self.img)
        des = self.orb.extract_descriptors(self.img, kps)

        self.assertTrue(len(des) > 0)
        self.assertTrue(len(kps), len(des))

class LKFeatureTrackerTest(unittest.TestCase):
    def setUp(self):
        detector = FAST(threshold=150)
        self.tracker = LKFeatureTracker(detector)

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
                cv2.imshow("VO Sequence " + self.data.sequence, self.img[index])
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


class FeatureTrackerTest(unittest.TestCase):
    def setUp(self):
        self.tracker = FeatureTracker()

        data_path = join(test.TEST_DATA_PATH, "vo")
        self.img0 = cv2.imread(join(data_path, "0.png"))
        self.img1 = cv2.imread(join(data_path, "1.png"))
        self.img2 = cv2.imread(join(data_path, "2.png"))
        self.img3 = cv2.imread(join(data_path, "3.png"))

        self.img = []
        for i in range(10):
            img_filename = "%d.png" % i
            self.img.append(cv2.imread(join(data_path, img_filename)))

    def test_initialize(self):
        self.tracker.initialize(self.img0)

        self.assertTrue(self.tracker.img_ref is not None)
        self.assertTrue(self.tracker.fea_ref is not None)
        self.assertTrue(len(self.tracker.fea_ref) > 0)
        self.assertEqual(self.tracker.counter_frame_id, 0)

    def test_detect(self):
        features = self.tracker.detect(self.img0)
        self.assertTrue(len(features) > 0)
        self.assertEqual(self.tracker.counter_frame_id, 0)

    def test_match(self):
        # Obtain features in both frames
        self.tracker.initialize(self.img0)
        f1 = self.tracker.detect(self.img1)

        # Perform matching
        matches, f0, f1 = self.tracker.match(f1)

        nb_matched = len(matches)
        nb_unmatched = len(f0) - nb_matched
        self.assertTrue(nb_matched < len(f0))
        self.assertEqual(len(self.tracker.unmatched), nb_unmatched)
        self.assertEqual(len(self.tracker.fea_ref), nb_matched)

        # Show matches
        # debug = True
        debug = False
        if debug:
            match_img = self.tracker.draw_matches(self.img0, self.img1,
                                                  f0, f1, matches)
            cv2.imshow("Matches", match_img)
            cv2.waitKey(0)

    def test_add_track(self):
        feature1 = Feature(np.zeros((2, 1)), 21, np.zeros((1, 21)))
        feature2 = Feature(np.zeros((2, 1)), 21, np.zeros((1, 21)))

        self.tracker.add_track(feature1, feature2)
        self.assertEqual(len(self.tracker.tracks_tracking), 1)
        self.assertEqual(len(self.tracker.tracks_lost), 0)
        self.assertEqual(len(self.tracker.tracks_buffer), 1)
        self.assertEqual(self.tracker.counter_track_id, 0)

    def test_remove_track(self):
        # Add feature track
        feature1 = Feature(np.zeros((2, 1)), 21, np.zeros((1, 21)))
        feature2 = Feature(np.zeros((2, 1)), 21, np.zeros((1, 21)))

        self.tracker.add_track(feature1, feature2)
        self.tracker.remove_track(0)

        self.assertEqual(len(self.tracker.tracks_tracking), 0)
        self.assertEqual(len(self.tracker.tracks_lost), 0)
        self.assertEqual(len(self.tracker.tracks_buffer), 0)
        self.assertEqual(self.tracker.counter_track_id, 0)

        # Add and remove feature track but this time mark as lost
        feature1 = Feature(np.zeros((2, 1)), 21, np.zeros((1, 21)))
        feature2 = Feature(np.zeros((2, 1)), 21, np.zeros((1, 21)))

        self.tracker.add_track(feature1, feature2)
        self.tracker.remove_track(1, True)

        self.assertEqual(len(self.tracker.tracks_tracking), 0)
        self.assertEqual(len(self.tracker.tracks_lost), 1)
        self.assertEqual(len(self.tracker.tracks_buffer), 1)
        self.assertEqual(self.tracker.counter_track_id, 1)

    def test_update_track(self):
        feature1 = Feature(np.zeros((2, 1)), 21, np.zeros((1, 21)))
        feature2 = Feature(np.zeros((2, 1)), 21, np.zeros((1, 21)))
        feature3 = Feature(np.zeros((2, 1)), 21, np.zeros((1, 21)))

        self.tracker.counter_frame_id = 1
        self.tracker.add_track(feature1, feature2)

        self.tracker.counter_frame_id = 2
        self.tracker.update_track(0, feature3)

        track = self.tracker.tracks_buffer[0]
        self.assertEqual(self.tracker.counter_frame_id, track.frame_end)
        self.assertEqual(0, track.frame_start)
        self.assertEqual(3, track.tracked_length())

    def test_process_matches(self):
        # Obtain features from images
        # self.tracker.debug_mode = True
        self.tracker.initialize(self.img0)
        f1 = self.tracker.detect(self.img1)

        # First match
        matches, f0, f1 = self.tracker.match(f1)
        self.tracker.process_matches(matches, f0, f1)

        self.assertTrue(len(self.tracker.tracks_tracking) > 0)
        self.assertEqual(len(self.tracker.tracks_tracking), len(matches))
        self.assertEqual(len(self.tracker.tracks_lost), 0)
        self.assertEqual(len(self.tracker.tracks_buffer), len(matches))

        track0 = self.tracker.tracks_buffer[0]
        self.assertEqual(track0.frame_start, 0)
        self.assertEqual(track0.frame_end, 1)
        self.assertEqual(track0.tracked_length(), 2)

        # Show matches
        # debug = True
        debug = False
        if debug:
            match_img = self.tracker.draw_matches(self.img0, self.img1,
                                                  f0, f1, matches)
            cv2.imshow("Matches", match_img)
            cv2.waitKey(0)

        # Second match
        f2 = self.tracker.detect(self.img2)
        matches, f1, f2 = self.tracker.match(f2)
        self.tracker.process_matches(matches, f1, f2)

        self.assertTrue(len(self.tracker.tracks_tracking) > 0)
        self.assertEqual(len(self.tracker.tracks_tracking), len(matches))
        self.assertTrue(len(self.tracker.tracks_lost) > 0)

        for i in range(len(self.tracker.tracks_lost)):
            track_id = self.tracker.tracks_lost[i]
            track_lost = self.tracker.tracks_buffer[track_id]

            current_frame_id = self.tracker.counter_frame_id
            tracked_length = track_lost.tracked_length()
            expected_frame_start = current_frame_id - tracked_length
            expected_frame_end = current_frame_id - 1
            self.assertEqual(track_lost.frame_start, expected_frame_start)
            self.assertEqual(track_lost.frame_end, expected_frame_end)

        # Show matches
        # debug = True
        debug = False
        if debug:
            match_img = self.tracker.draw_matches(self.img1, self.img2,
                                                  f1, f2, matches)
            cv2.imshow("Matches", match_img)
            cv2.waitKey(0)

        # Third match
        self.tracker.remove_lost_tracks()
        f3 = self.tracker.detect(self.img3)
        matches, f2, f3 = self.tracker.match(f3)
        self.tracker.process_matches(matches, f2, f3)

        self.assertTrue(len(self.tracker.tracks_tracking) > 0)
        self.assertEqual(len(self.tracker.tracks_tracking), len(matches))
        self.assertTrue(len(self.tracker.tracks_lost) > 0)

        for i in range(len(self.tracker.tracks_lost)):
            track_id = self.tracker.tracks_lost[i]
            track_lost = self.tracker.tracks_buffer[track_id]

            current_frame_id = self.tracker.counter_frame_id
            tracked_length = track_lost.tracked_length()
            expected_frame_start = current_frame_id - tracked_length
            expected_frame_end = current_frame_id - 1
            self.assertEqual(track_lost.frame_start, expected_frame_start)
            self.assertEqual(track_lost.frame_end, expected_frame_end)

        # Show matches
        # debug = True
        debug = False
        if debug:
            match_img = self.tracker.draw_matches(self.img2, self.img3,
                                                  f2, f3, matches)
            cv2.imshow("Matches", match_img)
            cv2.waitKey(0)

        # Plot feature tracks
        # debug = True
        debug = False
        if debug:
            for track_id in self.tracker.tracks_tracking:
                track_x = []
                track_y = []
                for feature in self.tracker.tracks_buffer[track_id].track:
                    track_x.append(feature.pt[0])
                    track_y.append(feature.pt[1])
                plt.plot(track_x, track_y)
            plt.show()

    def test_draw_matches(self):
        # Obtain features
        self.tracker.initialize(self.img0)
        f0 = self.tracker.fea_ref
        f1 = self.tracker.detect(self.img1)

        # Perform matching
        matched, f0, f1 = self.tracker.match(f1)

        # Draw matches
        img = self.tracker.draw_matches(self.img0, self.img1,
                                        f0, f1,
                                        matched)

        # Show matches
        # debug = True
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
        for track_id in self.tracker.tracks_tracking:
            track_x = []
            track_y = []

            if track_id is not None:
                for feature in self.tracker.tracks_buffer[track_id].track:
                    track_x.append(feature.pt[0])
                    track_y.append(feature.pt[1])
                ax.plot(track_x, track_y)
                fig.canvas.draw()

    def test_update(self):
        tracker = FeatureTracker(nb_features=500)

        # Stats
        tracked = []
        storage = []

        # Loop through images
        index = 0
        nb_images = len(self.img)
        time_start = time.time()
        # tracker.debug_mode = True

        while index < nb_images:
            # Index out of bounds guard
            index = 0 if index < 0 else index

            # Feature tracker update
            tracker.update(self.img[index])
            tracks_lost = tracker.remove_lost_tracks()

            # Assert feature track frame start and ends
            for i in range(len(tracks_lost)):
                track = tracks_lost[i]

                current_frame_id = tracker.counter_frame_id
                tracked_length = track.tracked_length()
                expected_frame_start = current_frame_id - tracked_length
                expected_frame_end = current_frame_id - 1
                self.assertEqual(track.frame_start, expected_frame_start)
                self.assertEqual(track.frame_end, expected_frame_end)

            # Record tracker stats
            tracked.append(len(tracker.tracks_tracking))
            storage.append(len(tracker.tracks_buffer))

            # Display image
            if tracker.debug_mode:
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
        print("fps: ", float(index) / time_elapsed)

        self.plot_tracked(tracked)
        self.plot_storage(storage)
        plt.show()
