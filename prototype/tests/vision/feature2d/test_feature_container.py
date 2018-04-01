import unittest

import numpy as np

from prototype.vision.feature2d.feature import Feature
from prototype.vision.feature2d.feature_container import FeatureContainer


class FeatureTrackerTest(unittest.TestCase):
    def setUp(self):
        self.container = FeatureContainer()

    def test_add_track(self):
        feature1 = Feature(np.zeros((2, 1)), 21, np.zeros((1, 21)))
        feature2 = Feature(np.zeros((2, 1)), 21, np.zeros((1, 21)))

        self.container.add_track(1, feature1, feature2)
        self.assertEqual(len(self.container.tracking), 1)
        self.assertEqual(len(self.container.lost), 0)
        self.assertEqual(len(self.container.data), 1)
        self.assertEqual(self.container.counter_track_id, 0)

    def test_remove_track(self):
        # Add feature track
        feature1 = Feature(np.zeros((2, 1)), 21, np.zeros((1, 21)))
        feature2 = Feature(np.zeros((2, 1)), 21, np.zeros((1, 21)))

        self.container.add_track(1, feature1, feature2)
        self.container.remove_track(0)

        self.assertEqual(len(self.container.tracking), 0)
        self.assertEqual(len(self.container.lost), 0)
        self.assertEqual(len(self.container.data), 0)
        self.assertEqual(self.container.counter_track_id, 0)

        # Add and remove feature track but this time mark as lost
        feature1 = Feature(np.zeros((2, 1)), 21, np.zeros((1, 21)))
        feature2 = Feature(np.zeros((2, 1)), 21, np.zeros((1, 21)))

        self.container.add_track(1, feature1, feature2)
        self.container.remove_track(1, True)

        self.assertEqual(len(self.container.tracking), 0)
        self.assertEqual(len(self.container.lost), 1)
        self.assertEqual(len(self.container.data), 1)
        self.assertEqual(self.container.counter_track_id, 1)

    def test_update_track(self):
        feature1 = Feature(np.zeros((2, 1)), 21, np.zeros((1, 21)))
        feature2 = Feature(np.zeros((2, 1)), 21, np.zeros((1, 21)))
        feature3 = Feature(np.zeros((2, 1)), 21, np.zeros((1, 21)))

        self.container.counter_frame_id = 1
        self.container.add_track(1, feature1, feature2)

        self.container.counter_frame_id = 2
        self.container.update_track(2, 0, feature3)

        track = self.container.data[0]
        self.assertEqual(2, track.frame_end)
        self.assertEqual(0, track.frame_start)
        self.assertEqual(3, track.tracked_length())
