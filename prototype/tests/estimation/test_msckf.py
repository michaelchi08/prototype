import unittest

import cv2
import matplotlib.pylab as plt

import numpy as np
from numpy import eye as I
from numpy import dot
from numpy import array_equal

from prototype.data.kitti import RawSequence
from prototype.utils.utils import deg2rad
from prototype.utils.euler import rotz
from prototype.utils.transform import T_rdf_flu
from prototype.utils.gps import latlon_diff
from prototype.utils.linalg import skew
from prototype.utils.quaternion.jpl import quat2rot as C
from prototype.utils.quaternion.jpl import quatnormalize
from prototype.utils.quaternion.jpl import quat2euler
from prototype.utils.quaternion.jpl import euler2quat
from prototype.vision.common import focal_length
from prototype.vision.common import camera_intrinsics
from prototype.vision.camera_model import PinholeCameraModel
from prototype.vision.features import Keypoint
from prototype.vision.features import FeatureTrack
from prototype.vision.features import FeatureTracker

from prototype.estimation.msckf import IMUState
from prototype.estimation.msckf import CameraState
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
        self.imu_state = IMUState(
            np.array([0.0, 0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0])
        )

    def test_F(self):
        w_hat = np.array([0.0, 0.0, 0.0])
        q_hat = np.array([1.0, 0.0, 0.0, 0.0])
        a_hat = np.array([1.0, 2.0, 3.0])
        w_G = np.array([0.0, 0.0, 0.0])

        F = self.imu_state.F(w_hat, q_hat, a_hat, w_G)

        # -- First row --
        self.assertTrue(array_equal(F[0:3, 0:3], -skew(w_hat)))
        self.assertTrue(array_equal(F[0:3, 3:6], -np.ones((3, 3))))
        # -- Third Row --
        self.assertTrue(array_equal(F[6:9, 0:3], dot(-C(q_hat).T, skew(a_hat))))
        self.assertTrue(array_equal(F[6:9, 6:9], -2.0 * skew(w_G)))
        self.assertTrue(array_equal(F[6:9, 9:12], -C(q_hat).T))
        self.assertTrue(array_equal(F[6:9, 12:15], -skew(w_G)**2))
        # -- Fifth Row --
        self.assertTrue(array_equal(F[12:15, 6:9], np.ones((3, 3))))

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

    def test_J(self):
        # Setup
        cam_q_CI = np.array([0.0, 0.0, 0.0, 1.0])
        cam_p_IC = np.array([0.0, 0.0, 0.0])
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
        self.assertTrue(array_equal(J[3:6, 9:12], I(3)))


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
                           cam_model=self.cam_model)

    def create_test_case(self):
        # Camera states
        cam_states = []
        # -- Camera state 0
        p_G_C0 = np.array([0.0, 0.0, 0.0])
        q_C0_G = np.array([0.0, 0.0, 0.0, 1.0])
        cam_states.append(CameraState(q_C0_G, p_G_C0))
        # -- Camera state 1
        p_G_C1 = np.array([1.0, 0.0, 0.0])
        q_C1_G = np.array([0.0, 0.0, 0.0, 1.0])
        cam_states.append(CameraState(q_C1_G, p_G_C1))

        # Features
        landmark = np.array([1.0, 2.0, 10.0, 1.0])
        noise = np.array([0.2, 0.2, 0.0])
        R_C0_G = np.array(C(q_C0_G))
        R_C1_G = np.array(C(q_C1_G))
        kp1 = self.cam_model.project(landmark, R_C0_G, p_G_C0) + noise
        kp2 = self.cam_model.project(landmark, R_C1_G, p_G_C1) + noise
        kp1 = Keypoint(kp1[:2], 21)
        kp2 = Keypoint(kp2[:2], 21)
        track = FeatureTrack(0, 1, kp1, kp2)

        return (self.cam_model, track, cam_states, landmark)

    def plot_position(self, pos_true, pos_est):
        N = pos_est.shape[1]
        pos_true = pos_true[:, :N]
        pos_est = pos_est[:, :N]

        plt.plot(pos_true[0, :], pos_true[1, :],
                 color="red", label="Grouth truth")
        plt.plot(pos_est[0, :], pos_est[1, :],
                 color="green", marker="o", label="Estimated")

        plt.xlabel("East (m)")
        plt.ylabel("North (m)")
        plt.axis("equal")

    def plot_attitude(self, timestamps, att_true, att_est):
        N = att_est.shape[1]
        t = timestamps[:N]
        att_true = att_true[:, :N]
        att_est = att_est[:, :N]

        plt.subplot(311)
        plt.plot(t, att_true[0, :], color="red")
        plt.plot(t, att_est[0, :], color="blue")
        plt.title("Roll")

        plt.subplot(312)
        plt.plot(t, att_true[1, :], color="red")
        plt.plot(t, att_est[1, :], color="blue")
        plt.title("Pitch")

        plt.subplot(313)
        plt.plot(t, att_true[2, :], color="red")
        plt.plot(t, att_est[2, :], color="blue")
        plt.title("Yaw")
        plt.xlabel("Date Time")
        plt.ylabel("rad s^-1")

    def plot_sliding_window(self, cam_states, yaw0):
        x = []
        y = []
        for cam_state in cam_states:
            cam_pos = cam_state.p_G
            cam_pos = np.array([cam_pos[2], -cam_pos[0], -cam_pos[1]])
            cam_pos = dot(rotz(-yaw0), cam_pos)
            x.append(cam_pos[0, 0])
            y.append(cam_pos[1, 0])

        plt.plot(x, y, color="blue", marker="o", label="Camera Poses")
        plt.xlabel("East (m)")
        plt.ylabel("North (m)")
        plt.axis("equal")

    def test_H(self):
        # Generate test case
        data = self.create_test_case()
        (cam_model, track, track_cam_states, landmark) = data

        # test H
        p_G_f, k, r = self.msckf.estimate_feature(cam_model,
                                                  track,
                                                  track_cam_states)
        self.msckf.augment_state()
        self.msckf.augment_state()
        H_f_j, H_x_j = self.msckf.H(track, track_cam_states, p_G_f)

        print(H_f_j)

        self.assertEqual(H_f_j.shape, (4, 3))
        self.assertEqual(H_x_j.shape, (4, 33))

    def test_prediction_update(self):
        # Setup
        debug = False
        # debug = True
        data = RawSequence(RAW_DATASET, "2011_09_26", "0005")
        K = data.calib_cam2cam["K_00"].reshape((3, 3))
        cam_model = PinholeCameraModel(1242, 375, K)

        # Initialize MSCKF
        v0 = np.array([data.oxts[0]["vf"], data.oxts[0]["vl"], 0.0])
        yaw0 = data.oxts[0]["yaw"]
        msckf = MSCKF(n_g=0.001 * np.ones(3),
                      n_a=0.001 * np.ones(3),
                      n_wg=0.001 * np.ones(3),
                      n_wa=0.001 * np.ones(3),
                      imu_v_G=T_rdf_flu * v0,
                      cam_model=cam_model)

        # Data storage
        pos_est = np.zeros((3, 1))
        att_est = np.zeros((3, 1))
        rpy_est = np.zeros((3, 1))

        # Loop through data
        for i in range(1, len(data.oxts[:20])):
            # Accelerometer and gyroscope and dt measurements
            a_m = T_rdf_flu * data.get_accel_true(i)
            w_m = T_rdf_flu * data.get_ang_vel_true(i)
            dt = data.get_dt(i)

            # MSCKF prediction and measurement update
            msckf.prediction_update(a_m, -w_m, dt)
            msckf_pos = dot(rotz(-yaw0), np.array([msckf.imu_state.p_G[2],
                                                   -msckf.imu_state.p_G[0],
                                                   -msckf.imu_state.p_G[1]]))
            msckf_att = quat2euler(msckf.imu_state.q_IG)
            msckf_att = np.array([-msckf_att[2], msckf_att[0], msckf_att[1]])
            msckf_rpy = np.array([-msckf.imu_state.rpy[2],
                                   msckf.imu_state.rpy[0],
                                   msckf.imu_state.rpy[1]])

            # Store history
            pos_est = np.hstack((pos_est, msckf_pos))
            att_est = np.hstack((att_est, msckf_att))
            rpy_est = np.hstack((rpy_est, msckf_rpy))

        if debug:
            self.plot_position(data.get_local_pos_true(), pos_est)
            self.plot_attitude(data.timestamps, data.get_att_true(), rpy_est)
            self.plot_attitude(data.timestamps, data.get_att_true(), att_est)
            data.plot_gyroscope()
            data.plot_accelerometer()
            plt.show()

    def test_estimate_feature(self):
        # Generate test case
        data = self.create_test_case()
        (cam_model, track, track_cam_states, landmark) = data

        # Estimate feature
        p_G_f, k, r = self.msckf.estimate_feature(cam_model,
                                                  track,
                                                  track_cam_states)

        # Debug
        debug = False
        if debug:
            print("\nk:", k)
            print("landmark:\n", landmark)
            print("p_G_f:\n", p_G_f)

        # Assert
        self.assertTrue(k < 10)
        self.assertTrue(abs(p_G_f[0, 0] - landmark[0]) < 0.1)
        self.assertTrue(abs(p_G_f[1, 0] - landmark[1]) < 0.1)
        self.assertTrue(abs(p_G_f[2, 0] - landmark[2]) < 0.1)

    def test_augment_state(self):
        self.msckf.augment_state()

    def test_measurement_update(self):
        # Setup
        debug = True
        # debug = False
        data = RawSequence(RAW_DATASET, "2011_09_26", "0046")
        # data = RawSequence(RAW_DATASET, "2011_09_26", "0036")
        K = data.calib_cam2cam["K_00"].reshape((3, 3))
        cam_model = PinholeCameraModel(1242, 375, K)

        # Initialize MSCKF
        v0 = np.array([data.oxts[0]["vf"], data.oxts[0]["vl"], 0.0])
        # q0 = euler2quat(data.get_att_true(0))
        msckf = MSCKF(n_g=0.001 * np.ones(3),
                      n_a=0.001 * np.ones(3),
                      n_wg=0.001 * np.ones(3),
                      n_wa=0.001 * np.ones(3),
                      # imu_q_IG=T_rdf_flu * q0,
                      imu_v_G=T_rdf_flu * v0,
                      cam_model=cam_model,
                      plot_covar=True)

        # Initialize feature tracker
        img = cv2.imread(data.image_00_files[0])
        tracker = FeatureTracker()
        tracker.update(img)

        # Data storage
        pos_est = np.zeros((3, 1))
        att_est = np.zeros((3, 1))
        yaw0 = data.oxts[0]["yaw"]

        # Loop through data
        for i in range(1, len(data.oxts)):
        # for i in range(1, 30):
            # Track features
            img = cv2.imread(data.image_00_files[i])
            # tracker.update(img, True)
            tracker.update(img)
            tracks = tracker.remove_lost_tracks()

            # Accelerometer and gyroscope and dt measurements
            a_m = T_rdf_flu * data.get_accel_true(i)
            w_m = T_rdf_flu * data.get_ang_vel_true(i)
            dt = data.get_dt(i)

            # MSCKF prediction and measurement update
            msckf.prediction_update(a_m, -w_m, dt)
            # msckf.measurement_update(tracks)

            # Store position
            msckf_pos = dot(rotz(-yaw0), np.array([msckf.imu_state.p_G[2],
                                                   -msckf.imu_state.p_G[0],
                                                   -msckf.imu_state.p_G[1]]))
            msckf_att = quat2euler(msckf.imu_state.q_IG)
            msckf_att = np.array([-msckf_att[2], msckf_att[0], msckf_att[1]])

            # Store history
            pos_est = np.hstack((pos_est, msckf_pos))
            att_est = np.hstack((att_est, msckf_att))

            msckf.covar_plot.update(msckf.P_imu)
            # if msckf.cax is None:
            #     msckf.plot_covariance(True)
            # else:
            #     msckf.cax.set_data(msckf.P_imu)
            #     index = 0
            #     for i in range(msckf.imu_state.size()):
            #         for j in range(msckf.imu_state.size()):
            #             txt = msckf.txt[index]
            #             txt.set_text(str(round(msckf.P_imu[i][j], 0)))
            #             index += 1
            #     msckf.fig.canvas.draw()

            # Show image frame
            # if debug:
            #     print("Frame: ", i)
            #     cv2.imshow("image", img)
            #     cv2.waitKey(1)

        # Plot
        # if debug:
        #     plt.figure()
        #     self.plot_position(data.get_local_pos_true(), pos_est)
        #     self.plot_sliding_window(msckf.cam_states, yaw0)
        #     plt.legend(loc=0)
        #
        #     plt.figure()
        #     self.plot_attitude(data.timestamps, data.get_att_true(), att_est)
        #     plt.show()
