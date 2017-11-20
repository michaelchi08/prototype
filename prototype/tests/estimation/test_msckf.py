import os
import random
import unittest

import cv2
import matplotlib.pylab as plt

import numpy as np
from numpy import eye as I
from numpy import dot
from numpy import array_equal

import prototype.tests as test
from prototype.data.kitti import RawSequence
from prototype.utils.utils import deg2rad
from prototype.utils.euler import rotz
from prototype.utils.linalg import skew
from prototype.utils.linalg import skewsq
from prototype.utils.quaternion.jpl import quat2rot as C
from prototype.utils.quaternion.jpl import euler2quat
from prototype.utils.transform import T_camera_global
from prototype.utils.transform import T_global_camera
from prototype.vision.common import focal_length
from prototype.vision.common import camera_intrinsics
from prototype.vision.common import rand3dfeatures
from prototype.vision.camera_model import PinholeCameraModel
from prototype.vision.features import Keypoint
from prototype.vision.features import FeatureTrack
from prototype.vision.features import FeatureTracker
from prototype.vision.dataset import DatasetGenerator
from prototype.vision.dataset import DatasetFeatureEstimator

from prototype.estimation.msckf import IMUState
from prototype.estimation.msckf import CameraState
from prototype.estimation.msckf import FeatureEstimator
from prototype.estimation.msckf import MSCKF

# GLOBAL VARIABLE
RAW_DATASET = "/data/raw"


class CameraStateTest(unittest.TestCase):
    def test_init(self):
        p_G = np.array([1.0, 2.0, 3.0])
        q_CG = np.array([1.0, 0.0, 0.0, 0.0])
        cam = CameraState(q_CG, p_G)

        self.assertTrue(array_equal(p_G, cam.p_G.ravel()))
        self.assertTrue(array_equal(q_CG, cam.q_CG.ravel()))


class IMUStateTest(unittest.TestCase):
    def setUp(self):
        n_g = np.ones(3) * 0.01  # Gyro Noise
        n_a = np.ones(3) * 0.01  # Accel Noise
        n_wg = np.ones(3) * 0.01  # Gyro Random Walk Noise
        n_wa = np.ones(3) * 0.01  # Accel Random Walk Noise
        n_imu = np.block([n_g.ravel(),
                          n_wg.ravel(),
                          n_a.ravel(),
                          n_wa.ravel()]).reshape((12, 1))

        self.imu_state = IMUState(
            np.array([0.0, 0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            n_imu
        )

    def test_F(self):
        w_hat = np.array([1.0, 2.0, 3.0])
        q_hat = np.array([0.0, 0.0, 0.0, 1.0])
        a_hat = np.array([1.0, 2.0, 3.0])
        w_G = np.array([0.1, 0.1, 0.1])

        F = self.imu_state.F(w_hat, q_hat, a_hat, w_G)

        # -- First row --
        self.assertTrue(array_equal(F[0:3, 0:3], -skew(w_hat)))
        self.assertTrue(array_equal(F[0:3, 3:6], -np.ones((3, 3))))
        # -- Third Row --
        self.assertTrue(array_equal(
            F[6:9, 0:3], dot(-C(q_hat).T, skew(a_hat))))
        self.assertTrue(array_equal(F[6:9, 6:9], -2.0 * skew(w_G)))
        self.assertTrue(array_equal(F[6:9, 9:12], -C(q_hat).T))
        self.assertTrue(array_equal(F[6:9, 12:15], -skewsq(w_G)))
        # -- Fifth Row --
        self.assertTrue(array_equal(F[12:15, 6:9], np.ones((3, 3))))

        # Plot matrix
        # debug = True
        debug = False
        if debug:
            ax = plt.subplot(111)
            ax.matshow(F)
            plt.show()

    def test_G(self):
        q_hat = np.array([0.0, 0.0, 0.0, 1.0]).reshape((4, 1))
        G = self.imu_state.G(q_hat)

        # -- First row --
        self.assertTrue(array_equal(G[0:3, 0:3], -np.ones((3, 3))))
        # -- Second row --
        self.assertTrue(array_equal(G[3:6, 3:6], np.ones((3, 3))))
        # -- Third row --
        self.assertTrue(array_equal(G[6:9, 6:9], -C(q_hat).T))
        # -- Fourth row --
        self.assertTrue(array_equal(G[9:12, 9:12], np.ones((3, 3))))

        # Plot matrix
        # debug = True
        debug = False
        if debug:
            ax = plt.subplot(111)
            ax.matshow(G)
            plt.show()

    def test_J(self):
        # Setup
        cam_q_CI = np.array([0.0, 0.0, 0.0, 1.0])
        cam_p_IC = np.array([1.0, 1.0, 1.0])
        q_hat_IG = np.array([0.0, 0.0, 0.0, 1.0])
        N = 1
        J = self.imu_state.J(cam_q_CI, cam_p_IC, q_hat_IG, N)

        # Assert
        C_CI = C(cam_q_CI)
        C_IG = C(q_hat_IG)
        # -- First row --
        self.assertTrue(array_equal(J[0:3, 0:3], C_CI))
        # -- Second row --
        self.assertTrue(array_equal(J[3:6, 0:3], skew(dot(C_IG.T, cam_p_IC))))
        # -- Third row --
        self.assertTrue(array_equal(J[3:6, 12:15], I(3)))

        # Plot matrix
        # debug = True
        debug = False
        if debug:
            ax = plt.subplot(111)
            ax.matshow(J)
            plt.show()


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
        p_G_C1 = np.array([0.0, 1.0, 0.0])
        rpy_C1G = np.array([deg2rad(0.0), deg2rad(0.0), deg2rad(0.0)])
        q_C1G = euler2quat(rpy_C1G)
        C_C1G = C(q_C1G)

        # Features
        landmark = np.array([0.0, 0.0, 10.0])
        kp1 = self.cam_model.project(landmark, C_C0G, p_G_C0)[0:2]
        kp2 = self.cam_model.project(landmark, C_C1G, p_G_C1)[0:2]

        # Calculate rotation and translation of first and last camera states
        # -- Set camera 0 as origin, work out rotation and translation of
        # -- camera 1 relative to to camera 0
        C_C0C1 = dot(C_C0G, C_C1G.T)
        t_C0_C1C0 = dot(C_C0G, (p_G_C1 - p_G_C0))
        # -- Convert from pixel coordinates to image coordinates
        pt1 = self.cam_model.pixel2image(kp1)
        pt2 = self.cam_model.pixel2image(kp2)

        # Triangulate
        p_C1C0_G = self.estimator.triangulate(pt1, pt2, C_C0C1, t_C0_C1C0)

        # Assert
        self.assertTrue(np.allclose(p_C1C0_G.ravel(), landmark))

    def test_estimate_feature(self):
        nb_features = 100
        bounds = {
            "x": {"min": -1.0, "max": 1.0},
            "y": {"min": -1.0, "max": 1.0},
            "z": {"min": 5.0, "max": 10.0}
        }
        features = rand3dfeatures(nb_features, bounds)

        dt = 0.1
        p_G = np.array([0.0, 0.0, 0.0])
        v_G = np.array([0.0, 0.0, 0.1])
        rpy_G = np.array([0.0, 0.0, 0.0])

        # Setup camera states
        track_cam_states = []
        for i in range(10):
            p_G = p_G + v_G * dt
            rpy_G[1] = np.random.normal(0.0, 0.01)
            q_CG = euler2quat(rpy_G)
            track_cam_states.append(CameraState(q_CG, p_G))

        # Feature Track
        start = 0
        end = 10
        feature_idx = random.randint(0, features.shape[1] - 1)
        feature = features[:, feature_idx]

        R_C0G = C(track_cam_states[0].q_CG)
        p_G_C0 = track_cam_states[0].p_G
        R_C1G = C(track_cam_states[1].q_CG)
        p_G_C1 = track_cam_states[1].p_G

        kp1 = self.cam_model.project(feature, R_C0G, p_G_C0)
        kp2 = self.cam_model.project(feature, R_C1G, p_G_C1)
        kp1 = Keypoint(kp1.ravel()[:2], 21)
        kp2 = Keypoint(kp2.ravel()[:2], 21)
        track = FeatureTrack(start, end, kp1, kp2)

        for i in range(2, 10):
            R_CG = C(track_cam_states[i].q_CG)
            p_G_C = track_cam_states[i].p_G
            kp = self.cam_model.project(feature, R_CG, p_G_C)
            kp = Keypoint(kp.ravel()[:2], 21)
            track.update(i, kp)

        # Estimate feature
        p_G_f, k, r = self.estimator.estimate(self.cam_model,
                                                      track,
                                                      track_cam_states)

        # Debug
        # debug = False
        debug = True
        if debug:
            print("\nk:", k)
            print("feature:\n", feature)
            print("p_G_f:\n", p_G_f)

        # Assert
        # self.assertTrue(k < 10)
        self.assertTrue(abs(p_G_f[0, 0] - feature[0]) < 0.1)
        self.assertTrue(abs(p_G_f[1, 0] - feature[1]) < 0.1)
        self.assertTrue(abs(p_G_f[2, 0] - feature[2]) < 0.1)


class MSCKFTest(unittest.TestCase):
    def setUp(self):
        # Pinhole Camera model
        image_width = 640
        image_height = 480
        fov = 60
        fx, fy = focal_length(image_width, image_height, fov)
        cx, cy = (image_width / 2.0, image_height / 2.0)
        K = camera_intrinsics(fx, fy, cx, cy)
        self.cam_model = PinholeCameraModel(image_width, image_height, K)

        # MSCKF
        self.msckf = MSCKF(n_g=0.001 * np.ones(3),
                           n_a=0.001 * np.ones(3),
                           n_wg=0.001 * np.ones(3),
                           n_wa=0.001 * np.ones(3),
                           ext_p_IC=np.array([0.0, 0.0, 0.0]),
                           ext_q_CI=np.array([0.0, 0.0, 0.0, 1.0]),
                           cam_model=self.cam_model)

    def plot_position(self, pos_true, pos_est, cam_states, yaw0):
        N = pos_est.shape[1]
        pos_true = pos_true[:, :N]
        pos_est = pos_est[:, :N]

        # Figure
        plt.figure()
        plt.suptitle("Position")

        # Ground truth
        plt.plot(pos_true[0, :], pos_true[1, :],
                 color="red", marker="o", label="Grouth truth")

        # Estimated
        plt.plot(pos_est[0, :], pos_est[1, :],
                 color="green", marker="o", label="Estimated")

        # Sliding window
        cam_pos = []
        for cam_state in cam_states:
            cam_pos.append(T_global_camera * cam_state.p_G)
        cam_pos = np.array(cam_pos).reshape((len(cam_pos), 3)).T
        plt.plot(cam_pos[0, :], cam_pos[1, :],
                 color="blue", marker="o", label="Camera Poses")

        # Plot labels and legends
        plt.xlabel("East (m)")
        plt.ylabel("North (m)")
        plt.axis("equal")
        plt.legend(loc=0)

    def plot_velocity(self, timestamps, vel_true, vel_est):
        N = vel_est.shape[1]
        t = timestamps[:N]
        vel_true = vel_true[:, :N]
        vel_est = vel_est[:, :N]

        # Figure
        plt.figure()
        plt.suptitle("Velocity")

        # X axis
        plt.subplot(311)
        plt.plot(t, vel_true[0, :], color="red", label="Ground_truth")
        plt.plot(t, vel_est[0, :], color="blue", label="Estimate")

        plt.title("x-axis")
        plt.xlabel("Date Time")
        plt.ylabel("ms^-1")
        plt.legend(loc=0)

        # Y axis
        plt.subplot(312)
        plt.plot(t, vel_true[1, :], color="red", label="Ground_truth")
        plt.plot(t, vel_est[1, :], color="blue", label="Estimate")

        plt.title("y-axis")
        plt.xlabel("Date Time")
        plt.ylabel("ms^-1")
        plt.legend(loc=0)

        # Z axis
        plt.subplot(313)
        plt.plot(t, vel_true[2, :], color="red", label="Ground_truth")
        plt.plot(t, vel_est[2, :], color="blue", label="Estimate")

        plt.title("z-axis")
        plt.xlabel("Date Time")
        plt.ylabel("ms^-1")
        plt.legend(loc=0)

    def plot_attitude(self, timestamps, att_true, att_est):
        N = att_est.shape[1]
        t = timestamps[:N]
        att_true = att_true[:, :N]
        att_est = att_est[:, :N]

        # Figure
        plt.figure()
        plt.suptitle("Attitude")

        # X axis
        plt.subplot(311)
        plt.plot(t, att_true[0, :], color="red", label="Ground_truth")
        plt.plot(t, att_est[0, :], color="blue", label="Estimate")

        plt.title("x-axis")
        plt.legend(loc=0)
        plt.xlabel("Date Time")
        plt.ylabel("rad s^-1")

        # Y axis
        plt.subplot(312)
        plt.plot(t, att_true[1, :], color="red", label="Ground_truth")
        plt.plot(t, att_est[1, :], color="blue", label="Estimate")

        plt.title("y-axis")
        plt.legend(loc=0)
        plt.xlabel("Date Time")
        plt.ylabel("rad s^-1")

        # Z axis
        plt.subplot(313)
        plt.plot(t, att_true[2, :], color="red", label="Ground_truth")
        plt.plot(t, att_est[2, :], color="blue", label="Estimate")

        plt.title("z-axis")
        plt.legend(loc=0)
        plt.xlabel("Date Time")
        plt.ylabel("rad s^-1")

    # def test_H(self):
    #     # Generate test case
    #     data = self.create_test_case()
    #     (cam_model, track, track_cam_states, landmark) = data
    #
    #     # test H
    #     p_G_f, k, r = self.msckf.estimate_feature(cam_model,
    #                                               track,
    #                                               track_cam_states)
    #     self.msckf.augment_state()
    #     self.msckf.augment_state()
    #     H_f_j, H_x_j = self.msckf.H(track, track_cam_states, p_G_f)
    #
    #     self.assertEqual(H_f_j.shape, (4, 3))
    #     self.assertEqual(H_x_j.shape, (4, 33))
    #
    #     # Plot matrix
    #     # debug = True
    #     debug = False
    #     if debug:
    #         ax = plt.subplot(211)
    #         ax.matshow(H_f_j)
    #         ax = plt.subplot(212)
    #         ax.matshow(H_x_j)
    #         plt.show()

    def test_prediction_update(self):
        # Setup
        data = RawSequence(RAW_DATASET, "2011_09_26", "0005")
        K = data.calib_cam2cam["K_00"].reshape((3, 3))
        cam_model = PinholeCameraModel(1242, 375, K)

        # Initialize MSCKF
        v0 = data.get_inertial_velocity(0)
        q0 = euler2quat(data.get_attitude(0))
        msckf = MSCKF(n_g=1e-6 * np.ones(3),
                      n_a=1e-6 * np.ones(3),
                      n_wg=1e-6 * np.ones(3),
                      n_wa=1e-6 * np.ones(3),
                      imu_q_IG=q0,
                      imu_v_G=v0,
                      cam_model=cam_model,
                      ext_p_IC=np.array([0.0, 0.0, 0.0]),
                      ext_q_CI=np.array([0.0, 0.0, 0.0, 1.0]))

        # Loop through data
        for i in range(1, len(data.oxts)):
            # MSCKF prediction and measurement update
            a_m, w_m = data.get_imu_measurements(i)
            dt = data.get_dt(i)
            msckf.prediction_update(a_m, w_m, dt)
            msckf.augment_state()

        # Plot
        # debug = True
        debug = False
        if debug:
            # Position
            self.plot_position(data.get_local_position(),
                               msckf.pos_est,
                               msckf.cam_states,
                               data.oxts[0]["yaw"])

            # Velocity
            self.plot_velocity(data.get_timestamps(),
                               data.get_inertial_velocity(),
                               msckf.vel_est)

            # Attitude
            self.plot_attitude(data.get_timestamps(),
                               data.get_attitude(),
                               msckf.att_est)

            # data.plot_accelerometer()
            # data.plot_gyroscope()
            plt.show()

    def test_prediction_update2(self):
        # Setup
        dataset = DatasetGenerator()

        # Initialize MSCKF
        v0 = dataset.v_B
        q0 = euler2quat(np.zeros((3, 1)))
        msckf = MSCKF(n_g=1e-6 * np.ones(3),
                      n_a=1e-6 * np.ones(3),
                      n_wg=1e-6 * np.ones(3),
                      n_wa=1e-6 * np.ones(3),
                      imu_q_IG=q0,
                      imu_v_G=v0,
                      cam_model=dataset.cam_model,
                      ext_p_IC=np.array([0.0, 0.0, 0.0]),
                      ext_q_CI=np.array([0.0, 0.0, 0.0, 1.0]))

        # Loop through data
        for i in range(1, 100):
            # MSCKF prediction and measurement update
            a_m, w_m = dataset.step()
            dt = dataset.dt
            msckf.prediction_update(a_m, w_m, dt)

        # Plot
        debug = True
        # debug = False
        if debug:
            # Position
            self.plot_position(dataset.pos_true,
                               msckf.pos_est,
                               msckf.cam_states,
                               dataset.rpy_true[0][2])

            # Velocity
            self.plot_velocity(dataset.time_true,
                               dataset.vel_true,
                               msckf.vel_est)

            # Attitude
            self.plot_attitude(dataset.time_true,
                               dataset.rpy_true,
                               msckf.att_est)

            # data.plot_accelerometer()
            # data.plot_gyroscope()
            plt.show()

    def test_augment_state(self):
        self.msckf.augment_state()

        # Plot matrix
        # debug = True
        debug = False
        if debug:
            ax = plt.subplot(111)
            ax.matshow(self.msckf.P())
            plt.show()

    def test_measurement_update(self):
        # Setup
        # data = RawSequence(RAW_DATASET, "2011_09_26", "0005")
        data = RawSequence(RAW_DATASET, "2011_09_26", "0046")
        # data = RawSequence(RAW_DATASET, "2011_09_26", "0036")
        K = data.calib_cam2cam["K_00"].reshape((3, 3))
        cam_model = PinholeCameraModel(1242, 375, K)

        # Initialize MSCKF
        v0 = data.get_inertial_velocity(0)
        q0 = euler2quat(data.get_attitude(0))
        msckf = MSCKF(n_g=1e-6 * np.ones(3),
                      n_a=1e-6 * np.ones(3),
                      n_wg=1e-6 * np.ones(3),
                      n_wa=1e-6 * np.ones(3),
                      imu_q_IG=q0,
                      imu_v_G=v0,
                      cam_model=cam_model,
                      ext_q_CI=np.array([0.0, 0.0, 0.0, 1.0]),
                      # ext_p_IC=np.array([1.08, -0.32, 0.0]),
                      ext_p_IC=np.zeros((3, 1)),
                      # plot_covar=True)
                      plot_covar=False)

        # Initialize feature tracker
        img = cv2.imread(data.image_00_files[0])
        tracker = FeatureTracker()
        tracker.update(img)

        # Loop through data
        for i in range(1, len(data.oxts)):
        # for i in range(1, 30):
            # Track features
            img = cv2.imread(data.image_00_files[i])
            # tracker.update(img, True)
            tracker.update(img)
            tracks = tracker.remove_lost_tracks()

            # Accelerometer and gyroscope and dt measurements
            a_m, w_m = data.get_imu_measurements(i)
            dt = data.get_dt(i)

            # MSCKF prediction and measurement update
            msckf.prediction_update(a_m, w_m, dt)
            msckf.measurement_update(tracks)
            msckf.update_plots()

        # Plot
        debug = True
        # debug = False
        if debug:
            # Position
            self.plot_position(data.get_local_position(),
                               msckf.pos_est,
                               msckf.cam_states,
                               data.oxts[0]["yaw"])

            # Velocity
            self.plot_velocity(data.get_timestamps(),
                               data.get_inertial_velocity(),
                               msckf.vel_est)

            # Attitude
            self.plot_attitude(data.get_timestamps(),
                               data.get_attitude(),
                               msckf.att_est)

            # data.plot_accelerometer()
            # data.plot_gyroscope()
            plt.show()

    def test_measurement_update2(self):
        # Setup
        dataset = DatasetGenerator()

        # Initialize MSCKF
        v0 = dataset.v_B
        q0 = euler2quat(np.zeros((3, 1)))
        msckf = MSCKF(n_g=1e-6 * np.ones(3),
                      n_a=1e-6 * np.ones(3),
                      n_wg=1e-6 * np.ones(3),
                      n_wa=1e-6 * np.ones(3),
                      imu_q_IG=q0,
                      imu_v_G=v0,
                      cam_model=dataset.cam_model,
                      ext_p_IC=np.array([0.0, 0.0, 0.0]),
                      ext_q_CI=np.array([0.0, 0.0, 0.0, 1.0]),
                      feature_estimator=DatasetFeatureEstimator())

        # Loop through data
        for i in range(1, 200):
            # Prediction update
            a_m, w_m = dataset.step()
            dt = dataset.dt
            msckf.prediction_update(a_m, w_m, dt)

            # Measurement update
            tracks = dataset.remove_lost_tracks()
            msckf.measurement_update(tracks)

            print("frame: %d" % i)

        # Plot
        debug = True
        # debug = False
        if debug:
            # Position
            self.plot_position(dataset.pos_true,
                               msckf.pos_est,
                               msckf.cam_states,
                               dataset.rpy_true[0][2])

            # Velocity
            self.plot_velocity(dataset.time_true,
                               dataset.vel_true,
                               msckf.vel_est)

            # Attitude
            self.plot_attitude(dataset.time_true,
                               dataset.rpy_true,
                               msckf.att_est)

            # data.plot_accelerometer()
            # data.plot_gyroscope()
            plt.show()
