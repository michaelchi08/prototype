import unittest

import cv2
import matplotlib.pylab as plt

import numpy as np

# import prototype.tests as test
from prototype.data.kitti import RawSequence
from prototype.utils.quaternion.jpl import euler2quat
from prototype.vision.common import focal_length
from prototype.vision.common import camera_intrinsics
from prototype.vision.camera_model import PinholeCameraModel
from prototype.vision.features import FeatureTracker
from prototype.vision.dataset import DatasetGenerator
from prototype.vision.dataset import DatasetFeatureEstimator

from prototype.estimation.msckf.msckf import MSCKF

# GLOBAL VARIABLE
RAW_DATASET = "/data/raw"


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

    def plot_position(self, pos_true, pos_est, cam_states):
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
                 color="blue", marker="o", label="Estimated")

        # Sliding window
        cam_pos = []
        for cam_state in cam_states:
            cam_pos.append(cam_state.p_G)
        cam_pos = np.array(cam_pos).reshape((len(cam_pos), 3)).T
        plt.plot(cam_pos[0, :], cam_pos[1, :],
                 color="green", marker="o", label="Camera Poses")

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
                      ext_q_CI=np.array([0.5, -0.5, 0.5, -0.5]))

        # Loop through data
        for i in range(1, len(data.oxts)):
            # MSCKF prediction and measurement update
            a_m, w_m = data.get_imu_measurements(i)
            dt = data.get_dt(i)
            msckf.prediction_update(a_m, w_m, dt)
            msckf.augment_state()

        # Plot
        debug = True
        # debug = False
        if debug:
            # Position
            self.plot_position(data.get_local_position(),
                               msckf.pos_est,
                               msckf.cam_states)

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
        # data = RawSequence(RAW_DATASET, "2011_09_26", "0001")
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
                      # ext_q_CI=np.array([0.0, 0.0, 0.0, 1.0]),
                      # ext_p_IC=np.zeros((3, 1)),
                      ext_q_CI=np.array([0.49921, -0.49657, 0.50291, -0.50129]),
                      ext_p_IC=np.array([1.08, -0.32, 0.0]),
                      # plot_covar=True)
                      plot_covar=False)

        # Initialize feature tracker
        img = cv2.imread(data.image_00_files[0])
        tracker = FeatureTracker()
        tracker.update(img)

        # Loop through data
        # for i in range(1, len(data.oxts)):
        for i in range(1, 30):
            print("frame %d" % i)
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
                               msckf.cam_states)

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
        dataset = DatasetGenerator(dt=0.1)

        # Initialize MSCKF
        v0 = dataset.v_B
        q0 = euler2quat(np.zeros((3, 1)))
        msckf = MSCKF(n_g=0.01 * np.ones(3),
                      n_a=0.01 * np.ones(3),
                      n_wg=0.01 * np.ones(3),
                      n_wa=0.01 * np.ones(3),
                      imu_q_IG=q0,
                      imu_v_G=v0,
                      cam_model=dataset.cam_model,
                      ext_p_IC=np.array([0.0, 0.0, 0.0]),
                      ext_q_CI=np.array([0.5, -0.5, 0.5, -0.5]),
                      feature_estimator=DatasetFeatureEstimator())

        # Loop through data
        for i in range(1, 20):
            print("frame: %d" % i)

            # Prediction update
            a_m, w_m = dataset.step()
            dt = dataset.dt
            msckf.prediction_update(a_m, w_m, dt)

            # Measurement update
            tracks = dataset.remove_lost_tracks()
            msckf.measurement_update(tracks)

        # Plot
        debug = True
        # debug = False
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
