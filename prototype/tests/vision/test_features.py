import sys
import unittest
from os.path import join

import cv2

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


# class LKFeatureTrackerTest(unittest.TestCase):
#     def setUp(self):
#         detector = FASTDetector(threshold=150)
#         self.tracker = LKFeatureTracker(detector)
#         self.data = VOSequence(VO_DATA_PATH, "00")
#
#     def test_detect(self):
#         img = cv2.imread(self.data.image_0_files[0])
#         self.tracker.detect(img)
#
#         self.assertTrue(len(self.tracker.tracks) > 10)
#         self.assertEqual(self.tracker.track_id,
#                          len(self.tracker.tracks))
#         self.assertEqual(self.tracker.track_id,
#                          len(self.tracker.tracks_alive))
#
#     def test_last_keypoints(self):
#         img = cv2.imread(self.data.image_0_files[0])
#         self.tracker.detect(img)
#         keypoints = self.tracker.last_keypoints()
#
#         self.assertEqual(len(self.tracker.tracks_alive), keypoints.shape[0])
#         self.assertEqual(2, keypoints.shape[1])
#
#     def test_track_features(self):
#         img1 = cv2.imread(self.data.image_0_files[0])
#         img2 = cv2.imread(self.data.image_0_files[1])
#
#         self.tracker.detect(img1)
#         tracks_alive_before = len(self.tracker.tracks_alive)
#
#         self.tracker.track_features(img1, img2)
#         tracks_alive_after = len(self.tracker.tracks_alive)
#
#         self.assertTrue(tracks_alive_after < tracks_alive_before)
#
#     def test_draw_tracks(self):
#         debug = False
#         img1 = cv2.imread(self.data.image_0_files[0])
#         img2 = cv2.imread(self.data.image_0_files[1])
#
#         self.tracker.detect(img1)
#         self.tracker.track_features(img1, img2)
#         self.tracker.draw_tracks(img2, debug)
#
#         if debug:
#             cv2.waitKey(1000000)
#
#     def test_update(self):
#         debug = False
#         tracks_tracked = []
#
#         # Loop through images
#         index = 0
#         while index <= len(self.data.image_0_files[:100]):
#             # Index out of bounds guard
#             index = 0 if index < 0 else index
#
#             # Feature tracker update
#             img = cv2.imread(self.data.image_0_files[index])
#             self.tracker.update(img, debug)
#             tracks_tracked.append(len(self.tracker.tracks_alive))
#
#             # Display image
#             if debug:
#                 cv2.imshow("VO Sequence " + self.data.sequence, img)
#                 key = cv2.waitKey(0)
#                 if key == ord('q'):  # Quit
#                     sys.exit(1)
#                 elif key == ord('p'):  # Previous image
#                     index -= 1
#                 else:
#                     index += 1
#             else:
#                 index += 1
#
#         if debug:
#             import matplotlib.pylab as plt
#             plt.plot(range(len(tracks_tracked)), tracks_tracked)
#             plt.show()
#             plt.clf()


class FeatureTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = FeatureTracker()
        self.data = VOSequence(VO_DATA_PATH, "00")

    def test_detect(self):
        img = cv2.imread(self.data.image_0_files[0])
        features = self.tracker.detect(img)
        self.assertTrue(len(features) > 10)

    def test_match(self):
        img0 = cv2.imread(self.data.image_0_files[0])
        img1 = cv2.imread(self.data.image_0_files[1])

        # Intialize tracker
        f0 = self.tracker.detect(img0)
        self.tracker.init_feature_tracks(f0)

        # Perform matching
        f1 = self.tracker.detect(img1)
        matches = self.tracker.match(self.tracker.tracks_alive, f1)

        self.assertTrue(len(matches) < len(f0))

    def test_draw_matches(self):
        img0 = cv2.imread(self.data.image_0_files[0])
        img1 = cv2.imread(self.data.image_0_files[1])

        # Intialize tracker
        f0 = self.tracker.detect(img0)
        self.tracker.init_feature_tracks(f0)

        # Perform matching
        f1 = self.tracker.detect(img1)
        matches = self.tracker.match(self.tracker.tracks_alive, f1)

        # Draw matches
        img = self.tracker.draw_matches(img0, img1,
                                        self.tracker.tracks_alive, f1,
                                        matches)

        # Show matches
        debug = True
        if debug:
            cv2.imshow("Matches", img)
            cv2.waitKey(0)

    def test_init_feature_tracks(self):
        img = cv2.imread(self.data.image_0_files[0])

        features = self.tracker.detect(img)
        self.tracker.keyframe = Keyframe(img, features)
        self.tracker.init_feature_tracks(self.tracker.keyframe.features)

        self.assertTrue(self.tracker.tracks_buffer, len(features))
        self.assertTrue(self.tracker.tracks_alive, len(features))
        self.assertEqual(self.tracker.track_id, len(features))

    def test_update_feature_tracks(self):
        img0 = cv2.imread(self.data.image_0_files[0])
        img1 = cv2.imread(self.data.image_0_files[1])
        img2 = cv2.imread(self.data.image_0_files[2])

        # Initialize feature tracks
        f0 = self.tracker.detect(img0)
        self.tracker.keyframe = Keyframe(img0, f0)
        self.tracker.init_feature_tracks(self.tracker.keyframe.features)

        print("tracks alive: ", len(self.tracker.tracks_alive))

        # Update feature tracks
        f1 = self.tracker.detect(img1)
        m1 = self.tracker.match(self.tracker.tracks_alive, f1)
        self.tracker.update_feature_tracks(f1, m1)

        print("tracks alive: ", len(self.tracker.tracks_alive))

        # Update feature tracks
        f2 = self.tracker.detect(img2)
        m2 = self.tracker.match(self.tracker.tracks_alive, f2)
        self.tracker.update_feature_tracks(f2, m2)

        print("tracks alive: ", len(self.tracker.tracks_alive))

        # # Plot feature tracks
        # import matplotlib.pylab as plt
        # for track in self.tracker.tracks_alive:
        #     track_x = []
        #     track_y = []
        #     for feature in track.track:
        #         track_x.append(feature.pt[0])
        #         track_y.append(feature.pt[1])
        #     plt.plot(track_x, track_y)
        # plt.show()

    def test_update(self):
        debug = True

        # Loop through images
        index = 0
        while index <= len(self.data.image_0_files[:1000]):
            # Index out of bounds guard
            index = 0 if index < 0 else index

            # Feature tracker update
            img = cv2.imread(self.data.image_0_files[index])
            self.tracker.update(img)

            # Display image
            if debug:
                # cv2.imshow("VO Sequence " + self.data.sequence, img)
                key = cv2.waitKey(0)
                if key == ord('q'):  # Quit
                    sys.exit(1)
                elif key == ord('p'):  # Previous image
                    index -= 1
                else:
                    index += 1
            else:
                index += 1
