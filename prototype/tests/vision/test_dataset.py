import os
import shutil
import unittest

import numpy as np
import matplotlib.pylab as plt

from prototype.vision.dataset import DatasetGenerator


class DatasetGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.save_dir = "/tmp/dataset_test"
        if os.path.isdir(self.save_dir):
            shutil.rmtree(self.save_dir)
        self.dataset = DatasetGenerator()

    def tearDown(self):
        if os.path.isdir(self.save_dir):
            shutil.rmtree(self.save_dir)

    def test_detect(self):
        # Setup
        dataset = DatasetGenerator(nb_features=100, debug_mode=True)

        # Test time step 1
        pos = np.array([0.0, 0.0, 0.0])
        rpy = np.array([0.0, 0.0, 0.0])
        dataset.detect(pos, rpy)
        tracks_prev = list(dataset.tracks_tracking)

        # Assert
        for track_id, track in dataset.tracks_buffer.items():
            self.assertEqual(track.track_id, track_id)
            self.assertEqual(track.frame_start, 0)
            self.assertEqual(track.frame_end, 0)
            self.assertEqual(len(track.track), 1)

        # Test time step 2
        pos = np.array([1.0, 0.0, 0.0])
        rpy = np.array([0.0, 0.0, 0.0])
        dataset.detect(pos, rpy)
        tracks_now = list(dataset.tracks_tracking)

        tracks_updated = set(tracks_prev).intersection(set(tracks_now))
        tracks_added = set(tracks_now) - set(tracks_prev)
        tracks_removed = set(tracks_prev) - set(tracks_now)

        # debug = True
        debug = False
        if debug:
            print("previous: ", tracks_prev)
            print("now: ", tracks_now)
            print("updated: ", tracks_updated)
            print("added: ", tracks_added)
            print("removed: ", tracks_removed)

        # Assert
        for track_id in tracks_updated:
            track = dataset.tracks_buffer[track_id]
            self.assertEqual(track.track_id, track_id)
            self.assertEqual(track.frame_start, 0)
            self.assertEqual(track.frame_end, 1)
            self.assertEqual(len(track.track), 2)

        for track_id in tracks_added:
            track = dataset.tracks_buffer[track_id]
            self.assertEqual(track.track_id, track_id)
            self.assertEqual(track.frame_start, 1)
            self.assertEqual(track.frame_end, 1)
            self.assertEqual(len(track.track), 1)

        for track_id in tracks_removed:
            self.assertTrue(track_id not in dataset.tracks_buffer)

    def test_step(self):
        # Step
        w_B_history = np.zeros((3, 1))
        a_B_history = np.zeros((3, 1))
        for i in range(30):
            (a_B, w_B) = self.dataset.step()
            a_B_history = np.hstack((a_B_history, a_B))
            w_B_history = np.hstack((w_B_history, w_B))

        # Plot
        debug = False
        debug = True
        if debug:
            plt.subplot(211)
            plt.plot(self.dataset.time_true, a_B_history[0, :], label="ax")
            plt.plot(self.dataset.time_true, a_B_history[1, :], label="ay")
            plt.plot(self.dataset.time_true, a_B_history[2, :], label="az")
            plt.legend(loc=0)

            plt.subplot(212)
            plt.plot(self.dataset.time_true, w_B_history[0, :], label="wx")
            plt.plot(self.dataset.time_true, w_B_history[1, :], label="wy")
            plt.plot(self.dataset.time_true, w_B_history[2, :], label="wz")
            plt.legend(loc=0)
            plt.show()


    # def test_simulate_test_data(self):
    #     self.dataset.simulate_test_data()

    # def test_generate_test_data(self):
    #     self.dataset.generate_test_data(self.save_dir)
