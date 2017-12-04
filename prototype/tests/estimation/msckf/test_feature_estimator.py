import time
import random
import unittest

import cv2
import numpy as np
from numpy import dot
import matplotlib.pylab as plt

from prototype.utils.utils import deg2rad
from prototype.utils.quaternion.jpl import quat2rot as C
from prototype.utils.quaternion.jpl import euler2quat
from prototype.utils.quaternion.jpl import quatlcomp
from prototype.utils.transform import T_camera_global
from prototype.utils.transform import T_global_camera
from prototype.utils.transform import R_global_camera
# from prototype.utils.transform import R_camera_global
from prototype.data.kitti import RawSequence
from prototype.vision.common import focal_length
from prototype.vision.common import camera_intrinsics
from prototype.vision.common import rand3dfeatures
from prototype.vision.camera_model import PinholeCameraModel
from prototype.vision.features import Keypoint
from prototype.vision.features import FeatureTrack
from prototype.vision.features import FeatureTracker

from prototype.estimation.msckf.camera_state import CameraState
from prototype.estimation.msckf.feature_estimator import FeatureEstimator


# GLOBAL VARIABLE
RAW_DATASET = "/data/raw"


class FeatureEstimatorTest(unittest.TestCase):
    def setUp(self):
        # Pinhole Camera model
        image_width = 640
        image_height = 480
        fov = 60
        fx, fy = focal_length(image_width, image_height, fov)
        cx, cy = (image_width / 2.0, image_height / 2.0)
        K = camera_intrinsics(fx, fy, cx, cy)
        self.cam_model = PinholeCameraModel(image_width, image_height, K)

        # Feature estimator
        self.estimator = FeatureEstimator()

    def test_triangulate(self):
        # Camera states
        # -- Camera state 0
        p_G_C0 = np.array([0.0, 0.0, 0.0])
        rpy_C0G = np.array([deg2rad(0.0), deg2rad(0.0), deg2rad(0.0)])
        q_C0G = euler2quat(rpy_C0G)
        C_C0G = C(q_C0G)
        # -- Camera state 1
        p_G_C1 = np.array([1.0, 1.0, 0.0])
        rpy_C1G = np.array([deg2rad(0.0), deg2rad(0.0), deg2rad(0.0)])
        q_C1G = euler2quat(rpy_C1G)
        C_C1G = C(q_C1G)

        # Features
        landmark = np.array([0.0, 0.0, 10.0])
        kp1 = self.cam_model.project(landmark, C_C0G, p_G_C0)[0:2]
        kp2 = self.cam_model.project(landmark, C_C1G, p_G_C1)[0:2]

        # Calculate rotation and translation of first and last camera states
        # -- Obtain rotation and translation from camera 0 to camera 1
        C_C0C1 = dot(C_C0G, C_C1G.T)
        t_C0_C1C0 = dot(C_C0G, (p_G_C1 - p_G_C0))
        # -- Convert from pixel coordinates to image coordinates
        pt1 = self.cam_model.pixel2image(kp1)
        pt2 = self.cam_model.pixel2image(kp2)

        # Triangulate
        p_C0_C1C0, r = self.estimator.triangulate(pt1, pt2, C_C0C1, t_C0_C1C0)

        # Assert
        self.assertTrue(np.allclose(p_C0_C1C0.ravel(), landmark))

    def test_estimate(self):
        nb_features = 100
        bounds = {
            "x": {"min": 5.0, "max": 10.0},
            "y": {"min": -1.0, "max": 1.0},
            "z": {"min": -1.0, "max": 1.0}
        }
        features = rand3dfeatures(nb_features, bounds)

        dt = 0.1
        p_G = np.array([0.0, 0.0, 0.0])
        v_G = np.array([0.1, 0.0, 0.0])
        q_CG = np.array([0.5, -0.5, 0.5, -0.5])

        # Setup camera states
        track_cam_states = []
        for i in range(10):
            p_G = p_G + v_G * dt
            track_cam_states.append(CameraState(i, q_CG, p_G))

        # Feature Track
        track_length = 10
        start = 0
        end = track_length

        for i in range(10):
            feature_idx = random.randint(0, features.shape[1] - 1)
            feature = T_camera_global * features[:, feature_idx]
            print("feature in global frame:",
                  (T_global_camera * feature).ravel())

            R_C0G = dot(R_global_camera, C(track_cam_states[0].q_CG))
            R_C1G = dot(R_global_camera, C(track_cam_states[1].q_CG))
            p_C_C0 = T_camera_global * track_cam_states[0].p_G
            p_C_C1 = T_camera_global * track_cam_states[1].p_G

            kp1 = self.cam_model.project(feature, R_C0G, p_C_C0)
            kp2 = self.cam_model.project(feature, R_C1G, p_C_C1)
            kp1 = Keypoint(kp1.ravel()[:2], 21)
            kp2 = Keypoint(kp2.ravel()[:2], 21)
            track = FeatureTrack(start, end, kp1, kp2)
            track.ground_truth = T_global_camera * feature

            for i in range(2, track_length):
                R_CG = dot(R_global_camera, C(track_cam_states[i].q_CG))
                p_C_Ci = T_camera_global * track_cam_states[i].p_G
                kp = self.cam_model.project(feature, R_CG, p_C_Ci)
                kp = Keypoint(kp.ravel()[:2], 21)
                # kp[0] += np.random.normal(0, 0.1)
                # kp[1] += np.random.normal(0, 0.1)
                track.update(i, kp)

            # Estimate feature
            p_G_f = self.estimator.estimate(self.cam_model,
                                            track,
                                            track_cam_states)
            print("estimation: ", p_G_f.ravel())
            print()

            # Debug
            # debug = False
            # debug = True
            # if debug:
            #     C_CG = C(track_cam_states[-1].q_CG)
            #     p_G_C = track_cam_states[-1].p_G
            #     p_C_f = dot(C_CG, (p_G_f - p_G_C))

            # Assert
            feature_G = T_global_camera * feature
            self.assertTrue(abs(p_G_f[0, 0] - feature_G[0]) < 0.1)
            self.assertTrue(abs(p_G_f[1, 0] - feature_G[1]) < 0.1)
            self.assertTrue(abs(p_G_f[2, 0] - feature_G[2]) < 0.1)

    def test_estimate2(self):
        # Load RAW KITTI dataset
        data = RawSequence(RAW_DATASET, "2011_09_26", "0001")
        # data = RawSequence(RAW_DATASET, "2011_09_26", "0046")
        # data = RawSequence(RAW_DATASET, "2011_09_26", "0005")
        K = data.calib_cam2cam["P_rect_00"].reshape((3, 4))[0:3, 0:3]
        cam_model = PinholeCameraModel(1242, 375, K)

        # Initialize feature tracker
        img = cv2.imread(data.image_00_files[0])
        tracker = FeatureTracker()
        tracker.update(img)

        # Setup plot
        fig = plt.figure()
        plt.ion()
        ax = fig.add_subplot(111)

        pos_data = data.get_local_position(0)
        pos_plot = ax.plot(pos_data[0], pos_data[1],
                           marker=".", color="blue")[0]

        features = None
        features_plot = None

        # ax.set_xlim([-60.0, 5.0])
        # ax.set_ylim([-60.0, 5.0])
        ax.set_xlim([0.0, 50.0])
        ax.set_ylim([-100.0, 0.0])
        fig.canvas.draw()
        plt.show(block=False)

        # Setup feature tracks
        tracks = []
        for i in range(1, 100):
            # Track features
            img = cv2.imread(data.image_00_files[i])
            tracker.update(img, True)
            if cv2.waitKey(1) == 113:
                exit(0)
            tracks = tracker.remove_lost_tracks()

            # Loop feature tracks
            p_G_f = None
            for track in tracks:
                if track.tracked_length() < 8:
                    continue

                # Setup feature track camera states
                track_cam_states = []
                for j in range(track.tracked_length()):
                    frame_id = track.frame_start + j

                    imu_q_IG = euler2quat(data.get_attitude(frame_id))
                    imu_p_G = data.get_local_position(frame_id)

                    ext_q_CI = np.array([0.5, -0.5, 0.5, -0.5])
                    cam_q_IG = dot(quatlcomp(ext_q_CI), imu_q_IG)
                    cam_p_G = imu_p_G

                    cam_state = CameraState(frame_id, cam_q_IG, cam_p_G)
                    track_cam_states.append(cam_state)

                # Estimate feature track
                p_G_f = self.estimator.estimate(cam_model,
                                                track,
                                                track_cam_states)
                print("estimation: ", p_G_f.ravel())

                if p_G_f is None:
                    C_CG = C(track_cam_states[-1].q_CG)
                    p_G_C = track_cam_states[-1].p_G
                    p_C_f = dot(C_CG, (p_G_f - p_G_C))

                    print("p_G_f: ", p_G_f.ravel())
                    print("p_C_f: ", p_C_f.ravel())
                print()

            # Plot
            pos = data.get_local_position(i)
            pos_data = np.hstack((pos_data, pos))

            if features is None and p_G_f is not None:
                features = p_G_f
                features_plot = ax.plot(p_G_f[0], p_G_f[1],
                                        marker="x", color="red", ls='')[0]
            elif p_G_f is not None:
                features = np.hstack((features, p_G_f))
                features_plot.set_xdata(features[0, :])
                features_plot.set_ydata(features[1, :])

            pos_plot.set_xdata(pos_data[0, :])
            pos_plot.set_ydata(pos_data[1, :])

            ax.relim()
            ax.autoscale_view(True, True, True)

            fig.canvas.draw()

        plt.pause(100000)

