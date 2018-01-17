import os
import shutil
import unittest

import numpy as np
import matplotlib.pylab as plt

from prototype.utils.transform import T_camera_global
from prototype.utils.transform import T_global_camera
from prototype.utils.utils import deg2rad
from prototype.utils.quaternion.jpl import quat2rot as C
from prototype.utils.quaternion.jpl import euler2quat
from prototype.vision.common import focal_length
from prototype.vision.common import camera_intrinsics
from prototype.vision.camera.camera_model import PinholeCameraModel
from prototype.vision.feature2d.keypoint import KeyPoint
from prototype.vision.feature2d.feature_track import FeatureTrack
from prototype.estimation.msckf.camera_state import CameraState
from prototype.vision.data.dataset import DatasetGenerator
from prototype.vision.data.dataset import DatasetFeatureEstimator


class DatasetFeatureEstimatorTest(unittest.TestCase):
    def test_estimate(self):
        estimator = DatasetFeatureEstimator()

        # Pinhole Camera model
        image_width = 640
        image_height = 480
        fov = 60
        fx, fy = focal_length(image_width, image_height, fov)
        cx, cy = (image_width / 2.0, image_height / 2.0)
        K = camera_intrinsics(fx, fy, cx, cy)
        cam_model = PinholeCameraModel(image_width, image_height, K)

        # Camera states
        track_cam_states = []
        # -- Camera state 0
        p_G_C0 = np.array([0.0, 0.0, 0.0])
        rpy_C0G = np.array([deg2rad(0.0), deg2rad(0.0), deg2rad(0.0)])
        q_C0G = euler2quat(rpy_C0G)
        C_C0G = C(q_C0G)
        track_cam_states.append(CameraState(0, q_C0G, p_G_C0))

        # -- Camera state 1
        p_G_C1 = np.array([1.0, 0.0, 0.0])
        rpy_C1G = np.array([deg2rad(0.0), deg2rad(0.0), deg2rad(0.0)])
        q_C1G = euler2quat(rpy_C1G)
        C_C1G = C(q_C1G)
        track_cam_states.append(CameraState(1, q_C1G, p_G_C1))

        # Feature track
        p_G_f = np.array([[0.0], [0.0], [10.0]])
        kp0 = KeyPoint(cam_model.project(p_G_f, C_C0G, p_G_C0)[0:2], 0)
        kp1 = KeyPoint(cam_model.project(p_G_f, C_C1G, p_G_C1)[0:2], 0)
        track = FeatureTrack(0, 1, kp0, kp1, ground_truth=p_G_f)
        estimate = estimator.estimate(cam_model, track, track_cam_states)

        self.assertTrue(np.allclose(p_G_f.ravel(), estimate.ravel(), atol=0.1))
        # self.assertTrue(np.allclose(estimator.r, np.zeros((4, 1))))


# class DatasetGeneratorTest(unittest.TestCase):
#     def setUp(self):
#         self.save_dir = "/tmp/dataset_test"
#         if os.path.isdir(self.save_dir):
#             shutil.rmtree(self.save_dir)
#         self.dataset = DatasetGenerator()
#
#     def tearDown(self):
#         if os.path.isdir(self.save_dir):
#             shutil.rmtree(self.save_dir)
#
#     def test_detect(self):
#         # Setup
#         dataset = DatasetGenerator(nb_features=100, debug_mode=True)
#
#         # Test time step 1
#         pos = np.array([0.0, 0.0, 0.0])
#         rpy = np.array([0.0, 0.0, 0.0])
#         dataset.detect(pos, rpy)
#         tracks_prev = list(dataset.tracks_tracking)
#
#         # Assert
#         for track_id, track in dataset.tracks_buffer.items():
#             self.assertEqual(track.track_id, track_id)
#             self.assertEqual(track.frame_start, 0)
#             self.assertEqual(track.frame_end, 0)
#             self.assertEqual(len(track.track), 1)
#
#         # Test time step 2
#         pos = np.array([1.0, 0.0, 0.0])
#         rpy = np.array([0.0, 0.0, 0.0])
#         dataset.detect(pos, rpy)
#         tracks_now = list(dataset.tracks_tracking)
#
#         tracks_updated = set(tracks_prev).intersection(set(tracks_now))
#         tracks_added = set(tracks_now) - set(tracks_prev)
#         tracks_removed = set(tracks_prev) - set(tracks_now)
#
#         # debug = True
#         debug = False
#         if debug:
#             print("previous: ", tracks_prev)
#             print("now: ", tracks_now)
#             print("updated: ", tracks_updated)
#             print("added: ", tracks_added)
#             print("removed: ", tracks_removed)
#
#         # Assert
#         for track_id in tracks_updated:
#             track = dataset.tracks_buffer[track_id]
#             self.assertEqual(track.track_id, track_id)
#             self.assertEqual(track.frame_start, 0)
#             self.assertEqual(track.frame_end, 1)
#             self.assertEqual(len(track.track), 2)
#
#         for track_id in tracks_added:
#             track = dataset.tracks_buffer[track_id]
#             self.assertEqual(track.track_id, track_id)
#             self.assertEqual(track.frame_start, 1)
#             self.assertEqual(track.frame_end, 1)
#             self.assertEqual(len(track.track), 1)
#
#         for track_id in tracks_removed:
#             self.assertTrue(track_id not in dataset.tracks_buffer)
#
#     def test_step(self):
#         # Step
#         a_B_history = self.dataset.a_B
#         w_B_history = self.dataset.w_B
#
#         for i in range(30):
#             (a_B, w_B) = self.dataset.step()
#             a_B_history = np.hstack((a_B_history, a_B))
#             w_B_history = np.hstack((w_B_history, w_B))
#
#         # Plot
#         debug = False
#         # debug = True
#         if debug:
#             plt.subplot(211)
#             plt.plot(self.dataset.time_true, a_B_history[0, :], label="ax")
#             plt.plot(self.dataset.time_true, a_B_history[1, :], label="ay")
#             plt.plot(self.dataset.time_true, a_B_history[2, :], label="az")
#             plt.legend(loc=0)
#
#             plt.subplot(212)
#             plt.plot(self.dataset.time_true, w_B_history[0, :], label="wx")
#             plt.plot(self.dataset.time_true, w_B_history[1, :], label="wy")
#             plt.plot(self.dataset.time_true, w_B_history[2, :], label="wz")
#             plt.legend(loc=0)
#             plt.show()
#
#     def test_estimate(self):
#         pass
#
#     # def test_simulate_test_data(self):
#     #     self.dataset.simulate_test_data()
#
#     # def test_generate_test_data(self):
#     #     self.dataset.generate_test_data(self.save_dir)
