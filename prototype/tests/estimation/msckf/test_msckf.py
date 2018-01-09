import unittest

import cv2
import matplotlib.pylab as plt

import numpy as np
from numpy import dot

# import prototype.tests as test
from prototype.utils.utils import deg2rad
from prototype.utils.quaternion.jpl import quat2rot as C
from prototype.data.kitti import RawSequence
from prototype.utils.quaternion.jpl import euler2quat
from prototype.utils.quaternion.jpl import quat2euler
from prototype.vision.common import focal_length
from prototype.vision.common import camera_intrinsics
from prototype.vision.camera_model import PinholeCameraModel
from prototype.vision.features import KeyPoint
from prototype.vision.features import FeatureTrack
from prototype.vision.features import FeatureTracker
from prototype.vision.dataset import DatasetGenerator
from prototype.vision.dataset import DatasetFeatureEstimator
from prototype.viz.plot_matrix import PlotMatrix

from prototype.estimation.msckf.msckf import MSCKF

# GLOBAL VARIABLE
RAW_DATASET = "/data/kitti/raw"


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
                           ext_q_CI=np.array([0.5, -0.5, 0.5, -0.5]),
                           cam_model=self.cam_model)

    def plot_position(self, pos_true, pos_est, cam_states):
        N = pos_est.shape[1]
        pos_true = pos_true[:, :N]
        pos_est = pos_est[:, :N]

        # Figure
        plt.figure()
        plt.suptitle("Position")

        # Ground truth
        plt.plot(pos_true[0, :], pos_true[1, :],
                 color="red", label="Grouth truth")
                 # color="red", marker="x", label="Grouth truth")

        # Estimated
        plt.plot(pos_est[0, :], pos_est[1, :],
                 color="blue", label="Estimated")
                 # color="blue", marker="o", label="Estimated")

        # Sliding window
        cam_pos = []
        for cam_state in cam_states:
            cam_pos.append(cam_state.p_G)
        cam_pos = np.array(cam_pos).reshape((len(cam_pos), 3)).T
        plt.plot(cam_pos[0, :], cam_pos[1, :],
                 color="green", label="Camera Poses")
                 # color="green", marker="o", label="Camera Poses")

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
        # Setup
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

    def test_init(self):
        self.assertEqual(self.msckf.N(), 1)
        self.assertTrue(self.msckf.P_cam is not None)
        self.assertTrue(self.msckf.P_imu_cam is not None)
        self.assertEqual(self.msckf.P_cam.shape, (6, 6))
        self.assertEqual(self.msckf.P_imu_cam.shape, (15, 6))

    def test_augment_state(self):
        self.msckf.augment_state()

        N = self.msckf.N()
        self.assertTrue(self.msckf.P_cam is not None)
        self.assertTrue(self.msckf.P_imu_cam is not None)
        self.assertEqual(self.msckf.P_cam.shape, (N * 6, N * 6))
        self.assertEqual(self.msckf.P_imu_cam.shape, (15, N * 6))
        self.assertEqual(self.msckf.N(), 2)

        self.assertTrue(np.array_equal(self.msckf.cam_states[0].q_CG,
                                       self.msckf.ext_q_CI))
        self.assertEqual(self.msckf.counter_frame_id, 2)

        # Plot matrix
        # debug = True
        debug = False
        if debug:
            ax = plt.subplot(111)
            ax.matshow(self.msckf.P())
            plt.show()

    def test_track_cam_states(self):
        # Setup feature track
        track_id = 0
        frame_id = 3
        data0 = KeyPoint(np.array([0.0, 0.0]), 21)
        data1 = KeyPoint(np.array([0.0, 0.0]), 21)
        track = FeatureTrack(track_id, frame_id, data0, data1)

        # Push dummy camera states into MSCKF
        self.msckf.augment_state()
        self.msckf.augment_state()
        self.msckf.augment_state()
        self.msckf.augment_state()

        # Test
        track_cam_states = self.msckf.track_cam_states(track)

        # Assert
        self.assertEqual(len(track_cam_states), track.tracked_length())
        self.assertEqual(track_cam_states[0].frame_id, track.frame_start)
        self.assertEqual(track_cam_states[1].frame_id, track.frame_end)

    def test_P(self):
        self.assertEqual(self.msckf.P().shape, (21, 21))

        # Plot matrix
        # debug = True
        debug = False
        if debug:
            ax = plt.subplot(111)
            ax.matshow(self.msckf.P())
            plt.show()

    def test_N(self):
        self.assertEqual(self.msckf.N(), 1)

    def test_H(self):
        # Setup feature track
        track_id = 0
        frame_id = 3
        data0 = KeyPoint(np.array([0.0, 0.0]), 21)
        data1 = KeyPoint(np.array([0.0, 0.0]), 21)
        track = FeatureTrack(track_id, frame_id, data0, data1)

        # Setup track cam states
        self.msckf.augment_state()
        self.msckf.augment_state()
        self.msckf.augment_state()
        track_cam_states = self.msckf.track_cam_states(track)

        # Feature position
        p_G_f = np.array([[1.0], [2.0], [3.0]])

        # Test
        H_f_j, H_x_j = self.msckf.H(track, track_cam_states, p_G_f)

        # Assert
        self.assertEqual(H_f_j.shape, (4, 3))
        self.assertEqual(H_x_j.shape, (4, 39))

        # Plot matrix
        # debug = True
        debug = False
        if debug:
            plt.matshow(H_f_j)
            plt.show()
            plt.matshow(H_x_j)
            plt.show()

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
                      ext_q_CI=np.array([0.5, -0.5, 0.5, -0.5]))

        # Setup state history storage and covariance plot
        pos_est = msckf.imu_state.p_G
        vel_est = msckf.imu_state.v_G
        att_est = quat2euler(msckf.imu_state.q_IG)

        # Loop through data
        for i in range(1, len(data.oxts)):
            # MSCKF prediction and measurement update
            a_m, w_m = data.get_imu_measurements(i)
            dt = data.get_dt(i)
            msckf.prediction_update(a_m, w_m, dt)
            msckf.augment_state()

            # Store history
            pos = msckf.imu_state.p_G
            vel = msckf.imu_state.v_G
            att = quat2euler(msckf.imu_state.q_IG)

            pos_est = np.hstack((pos_est, pos))
            vel_est = np.hstack((vel_est, vel))
            att_est = np.hstack((att_est, att))

        # Plot
        # debug = True
        debug = False
        if debug:
            # Position
            self.plot_position(data.get_local_position(),
                               pos_est,
                               msckf.cam_states)

            # Velocity
            self.plot_velocity(data.get_timestamps(),
                               data.get_inertial_velocity(),
                               vel_est)

            # Attitude
            self.plot_attitude(data.get_timestamps(),
                               data.get_attitude(),
                               att_est)

            # data.plot_accelerometer()
            # data.plot_gyroscope()
            plt.show()

    def test_prediction_update2(self):
        # Setup
        dataset = DatasetGenerator()

        # Initialize MSCKF
        v0 = dataset.vel
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
        for i in range(1, 30):
            # MSCKF prediction and measurement update
            a_m, w_m = dataset.step()
            dt = dataset.dt
            msckf.prediction_update(a_m, w_m, dt)

        # Plot
        # debug = True
        debug = False
        if debug:
            # Position
            self.plot_position(dataset.pos_true,
                               msckf.pos_est,
                               msckf.cam_states)

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

    def test_track_residuals(self):
        # Setup feature track
        track_id = 0
        frame_id = 3
        data0 = KeyPoint(np.array([320.0, 240.0]), 21)
        data1 = KeyPoint(np.array([320.0, 240.0]), 21)
        track = FeatureTrack(track_id, frame_id, data0, data1)

        # Setup track cam states
        self.msckf.augment_state()
        self.msckf.augment_state()
        self.msckf.augment_state()
        self.msckf.augment_state()
        track_cam_states = self.msckf.track_cam_states(track)

        # Setup feature
        p_f_G = np.array([[10.0], [0.0], [0.0]])

        # Test
        r_j = self.msckf.track_residuals(self.cam_model,
                                         track,
                                         track_cam_states,
                                         p_f_G)

        # Assert
        self.assertEqual(r_j.shape, (4, 1))
        self.assertTrue(np.allclose(r_j, np.zeros((4, 1))))

    def test_residualize_track(self):
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

        # Setup feature track
        track_id = 0
        frame_id = 1
        data1 = KeyPoint(kp1, 21)
        data2 = KeyPoint(kp2, 21)
        track = FeatureTrack(track_id, frame_id, data1, data2)

        # Setup track cam states
        self.msckf.augment_state()
        self.msckf.min_track_length = 2
        self.msckf.cam_states[0].p_G = p_G_C0.reshape((3, 1))
        self.msckf.cam_states[0].q_CG = q_C0G.reshape((4, 1))
        self.msckf.cam_states[1].p_G = p_G_C1.reshape((3, 1))
        self.msckf.cam_states[1].q_CG = q_C1G.reshape((4, 1))

        # Test
        # self.msckf.enable_ns_trick = False
        H_o_j, r_o_j, R_o_j = self.msckf.residualize_track(track)

        # plt.matshow(H_o_j)
        # plt.show()
        # plt.matshow(r_o_j)
        # plt.show()
        # plt.matshow(R_o_j)
        # plt.show()

    def test_measurement_update(self):
        # Setup
        dataset = DatasetGenerator(dt=0.1)

        # Initialize MSCKF
        q0 = euler2quat(np.zeros((3, 1)))
        v0 = dataset.vel
        msckf = MSCKF(n_g=0.001 * np.ones(3),
                      n_a=0.001 * np.ones(3),
                      n_wg=0.001 * np.ones(3),
                      n_wa=0.001 * np.ones(3),
                      imu_q_IG=q0,
                      imu_v_G=v0,
                      cam_model=dataset.cam_model,
                      ext_p_IC=np.array([0.0, 0.0, 0.0]),
                      ext_q_CI=np.array([0.5, -0.5, 0.5, -0.5]))
                      # feature_estimator=DatasetFeatureEstimator())

        # cov_plot = PlotMatrix(msckf.P())
        # plt.show(block=False)

        # Setup state history storage and covariance plot
        pos_est = msckf.imu_state.p_G
        vel_est = msckf.imu_state.v_G
        att_est = quat2euler(msckf.imu_state.q_IG)

        np.random.seed(0)

        # Loop through data
        for i in range(1, 100):
            print("frame: %d" % i)

            # Prediction update
            a_m, w_m = dataset.step()
            dt = dataset.dt
            msckf.prediction_update(a_m, w_m, dt)

            # Measurement update
            tracks = dataset.remove_lost_tracks()
            msckf.measurement_update(tracks)

            # cov_plot.update(msckf.P())

            pos = msckf.imu_state.p_G
            vel = msckf.imu_state.v_G
            att = quat2euler(msckf.imu_state.q_IG)

            pos_est = np.hstack((pos_est, pos))
            vel_est = np.hstack((vel_est, vel))
            att_est = np.hstack((att_est, att))

        # Plot
        # debug = True
        debug = False
        if debug:
            # Position
            self.plot_position(dataset.pos_true,
                               pos_est,
                               msckf.cam_states)

            # Velocity
            self.plot_velocity(dataset.time_true,
                               dataset.vel_true,
                               vel_est)

            # Attitude
            self.plot_attitude(dataset.time_true,
                               dataset.att_true,
                               att_est)

            # data.plot_accelerometer()
            # data.plot_gyroscope()
            plt.show()

    def test_measurement_update2(self):
        # Setup
        # data = RawSequence(RAW_DATASET, "2011_09_26", "0001")
        data = RawSequence(RAW_DATASET, "2011_09_26", "0005")
        # data = RawSequence(RAW_DATASET, "2011_09_26", "0046")
        # data = RawSequence(RAW_DATASET, "2011_09_26", "0036")
        K = data.calib_cam2cam["P_rect_00"].reshape((3, 4))[0:3, 0:3]
        cam_model = PinholeCameraModel(1242, 375, K)

        # Initialize MSCKF
        v0 = data.get_inertial_velocity(0)
        print(v0)
        exit(0)
        q0 = euler2quat(data.get_attitude(0))
        msckf = MSCKF(n_g=4e-2 * np.ones(3),
                      n_a=4e-2 * np.ones(3),
                      n_wg=1e-6 * np.ones(3),
                      n_wa=1e-6 * np.ones(3),
                      imu_q_IG=q0,
                      imu_v_G=v0,
                      cam_model=cam_model,
                      # ext_q_CI=np.array([0.0, 0.0, 0.0, 1.0]),
                      ext_p_IC=np.zeros((3, 1)),
                      ext_q_CI=np.array([0.49921, -0.49657, 0.50291, -0.50129]))
                      # ext_p_IC=np.array([1.08, -0.32, 0.0]))

        # Initialize feature tracker
        img = cv2.imread(data.image_00_files[0])
        tracker = FeatureTracker()
        tracker.update(img)

        # Setup state history storage and covariance plot
        pos_est = msckf.imu_state.p_G
        vel_est = msckf.imu_state.v_G
        att_est = quat2euler(msckf.imu_state.q_IG)

        # Loop through data
        for i in range(1, len(data.oxts)):
        # for i in range(1, 100):
            print("frame %d" % i)
            # Track features
            img = cv2.imread(data.image_00_files[i])
            # tracker.update(img, True)
            # key = cv2.waitKey(1)
            # if key == 113:
            #     exit(0)
            # tracker.update(img)
            tracks = tracker.remove_lost_tracks()

            # Accelerometer and gyroscope and dt measurements
            a_m, w_m = data.get_imu_measurements(i)
            dt = data.get_dt(i)

            # MSCKF prediction and measurement update
            msckf.prediction_update(a_m, w_m, dt)
            # msckf.measurement_update(tracks)

            # Store history
            pos = msckf.imu_state.p_G
            vel = msckf.imu_state.v_G
            att = quat2euler(msckf.imu_state.q_IG)

            pos_est = np.hstack((pos_est, pos))
            vel_est = np.hstack((vel_est, vel))
            att_est = np.hstack((att_est, att))

        # Plot
        debug = True
        # debug = False
        if debug:
            # Position
            self.plot_position(data.get_local_position(),
                               pos_est,
                               msckf.cam_states)

            # Velocity
            self.plot_velocity(data.get_timestamps(),
                               data.get_inertial_velocity(),
                               vel_est)

            # Attitude
            self.plot_attitude(data.get_timestamps(),
                               data.get_attitude(),
                               att_est)

            # data.plot_accelerometer()
            # data.plot_gyroscope()
            plt.show()
