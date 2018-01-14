import unittest
from os.path import join

import cv2
import numpy as np

import prototype.tests as test
from prototype.data.kitti import RawSequence
from prototype.vision.feature2d.common import draw_points
from prototype.vision.feature2d.klt_tracker import KLTTracker


class KLTTrackerTest(unittest.TestCase):
    def setUp(self):
        self.tracker = KLTTracker()

        data_path = join(test.TEST_DATA_PATH, "vo")
        self.img0 = cv2.imread(join(data_path, "0.png"))
        self.img1 = cv2.imread(join(data_path, "1.png"))
        self.img2 = cv2.imread(join(data_path, "2.png"))
        self.img3 = cv2.imread(join(data_path, "3.png"))

    def test_constructor(self):
        self.assertEqual(1000, self.tracker.nb_max_corners)
        self.assertEqual(0.01, self.tracker.quality_level)
        self.assertEqual(10, self.tracker.min_distance)

    def test_initialize(self):
        features = self.tracker.initialize(self.img0)
        self.assertEqual(0, self.tracker.counter_frame_id)
        self.assertTrue(len(features) > 0)

    def test_process_track(self):
        features = self.tracker.initialize(self.img0)

        # Test add track
        self.tracker.counter_frame_id += 1
        f0 = features[0]
        f1 = features[1]
        keeping = []
        self.tracker.process_track(1, f0, f1, keeping)
        self.assertEqual(1, len(self.tracker.features.tracking))
        self.assertEqual(0, len(self.tracker.features.lost))
        self.assertEqual(1, len(self.tracker.features.data))

        # Test update track
        self.tracker.counter_frame_id += 1
        f2 = features[2]
        f0 = self.tracker.features.data[0]
        keeping = []
        self.tracker.process_track(1, f0, f2, keeping)
        self.assertEqual(1, len(self.tracker.features.tracking))
        self.assertEqual(0, len(self.tracker.features.lost))
        self.assertEqual(1, len(self.tracker.features.data))
        self.assertEqual(3, len(f0.track))

        # Test remove track
        self.tracker.counter_frame_id += 1
        keeping = []
        self.tracker.process_track(0, f0, f2, keeping)
        self.assertEqual(0, len(self.tracker.features.tracking))
        self.assertEqual(1, len(self.tracker.features.lost))
        self.assertEqual(1, len(self.tracker.features.data))

    def test_track(self):
        features = self.tracker.initialize(self.img0)

        # Test track features
        self.tracker.counter_frame_id += 1
        self.tracker.img_cur = self.img0
        self.tracker.track(features)
        self.assertEqual(len(features), len(self.tracker.features.tracking))
        self.assertEqual(0, len(self.tracker.features.lost))
        self.assertEqual(len(features), len(self.tracker.features.data))

        # Test track features
        self.tracker.counter_frame_id += 1
        self.tracker.img_cur = self.img1
        self.tracker.track(self.tracker.features.fea_ref)

    def test_update(self):
        self.tracker.update(self.img0)

        # img = draw_features(self.img_cur, keeping)
        # cv2.imshow("Features", img)
        # cv2.waitKey(0)
