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
from prototype.estimation.msckf import CameraState
from prototype.estimation.msckf import MSCKF
from prototype.vision.common import focal_length
from prototype.vision.common import camera_intrinsics
from prototype.vision.camera_model import PinholeCameraModel
from prototype.vision.features import Keypoint
from prototype.vision.features import FeatureTrack
from prototype.vision.features import FeatureTracker

# GLOBAL VARIABLE
RAW_DATASET = "/data/raw"


class CameraStateTest(unittest.TestCase):
    def test_init(self):
        p_G = np.array([1.0, 2.0, 3.0])
        q_CG = np.array([1.0, 0.0, 0.0, 0.0])
        cam = CameraState(q_CG, p_G)

        self.assertTrue(array_equal(p_G, cam.p_G.ravel()))
        self.assertTrue(array_equal(q_CG, cam.q_CG.ravel()))


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

    def test_F(self):
        w_hat = np.array([0.0, 0.0, 0.0])
        q_hat = np.array([1.0, 0.0, 0.0, 0.0])
        a_hat = np.array([1.0, 2.0, 3.0])
        w_G = np.array([0.0, 0.0, 0.0])

        F = self.msckf.F(w_hat, q_hat, a_hat, w_G)

        # -- First row --
        self.assertTrue(array_equal(F[0:3, 0:3], -skew(w_hat)))
        self.assertTrue(array_equal(F[0:3, 3:6], np.ones((3, 3))))
        # -- Third Row --
        self.assertTrue(array_equal(F[6:9, 0:3], dot(-C(q_hat).T, skew(a_hat))))
        self.assertTrue(array_equal(F[6:9, 6:9], -2.0 * skew(w_G)))
        self.assertTrue(array_equal(F[6:9, 9:12], -C(q_hat).T))
        self.assertTrue(array_equal(F[6:9, 12:15], -skew(w_G)**2))
        # -- Fifth Row --
        self.assertTrue(array_equal(F[12:15, 6:9], np.ones((3, 3))))

    def test_G(self):
        q_hat = np.array([1.0, 0.0, 0.0, 0.0]).reshape((4, 1))
        G = self.msckf.G(q_hat)

        # -- First row --
        self.assertTrue(array_equal(G[0:3, 0:3], np.ones((3, 3))))
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
        J = self.msckf.J(cam_q_CI, cam_p_IC, q_hat_IG, N)

        # Assert
        C_CI = C(cam_q_CI)
        C_IG = C(q_hat_IG)
        # -- First row --
        self.assertTrue(array_equal(J[0:3, 0:3], C_CI))
        # -- Second row --
        self.assertTrue(array_equal(J[3:6, 0:3], skew(dot(C_IG.T, cam_p_IC))))
        # -- Third row --
        self.assertTrue(array_equal(J[3:6, 9:12], I(3)))

    def test_H(self):
        pass

    def test_prediction_update(self):
        a_m = np.array([[0.0], [0.0], [10.0]])
        w_m = np.array([[0.0], [0.0], [0.0]])
        dt = 0.1

        for i in range(10):
            self.msckf.prediction_update(a_m, w_m, dt)

    def test_estimate_feature(self):
        # Generate test case
        data = self.create_test_case()
        (cam_model, track, track_cam_states, landmark) = data

        # Estimate feature
        results = self.msckf.estimate_feature(cam_model,
                                              track,
                                              track_cam_states)
        p_G_f, k, r = results

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

    def test_calculate_track_residual(self):
        # Generate test case
        data = self.create_test_case()
        (cam_model, track, track_cam_states, landmark) = data

        # Estimate feature
        p_G_f, k, r = self.msckf.estimate_feature(cam_model,
                                                  track,
                                                  track_cam_states)

        # Calculate track residual
        debug = False
        residual = self.msckf.calculate_track_residual(cam_model,
                                                       track,
                                                       track_cam_states,
                                                       p_G_f,
                                                       debug)
        if debug:
            print("residual:", residual)

    def test_augment_state(self):
        self.msckf.augment_state()

    def test_measurement_update(self):
        # Setup
        debug = True
        data = RawSequence(RAW_DATASET, "2011_09_26", "0046")
        tracker = FeatureTracker()

        # MSCKF
        v0 = np.array([data.oxts[0]["vf"],
                       data.oxts[0]["vl"],
                       0.0])
        msckf = MSCKF(n_g=0.001 * np.ones(3),
                      n_a=0.001 * np.ones(3),
                      n_wg=0.001 * np.ones(3),
                      n_wa=0.001 * np.ones(3),
                      imu_v_G=v0,
                      cam_model=self.cam_model)

        # Home point
        lat_ref = data.oxts[0]['lat']
        lon_ref = data.oxts[0]['lon']
        N = len(data.oxts)

        # Data storage
        t = [data.timestamps[0]]
        ground_truth_x = []
        ground_truth_y = []
        msckf_x = [0.0]
        msckf_y = [0.0]
        msckf_vx = [data.oxts[0]["vf"]]
        msckf_vy = [data.oxts[0]["vl"]]

        yaw0 = data.oxts[0]["yaw"]
        # print("Init yaw: ", yaw0)

        # Loop through data
        t_prev = data.timestamps[0]
        for i in range(1, 5):
            # Calculate position relative to home point
            lat = data.oxts[i]['lat']
            lon = data.oxts[i]['lon']
            dist_N, dist_E = latlon_diff(lat_ref, lon_ref, lat, lon)

            # Track features
            img = cv2.imread(data.image_00_files[i])
            tracker.update(img)
            tracks = tracker.remove_lost_tracks()

            # Accelerometer and gyroscope measurements
            a_m = np.array([[data.oxts[i]["ax"]],
                            [data.oxts[i]["ay"]],
                            [data.oxts[i]["az"]]])
            w_m = np.array([[data.oxts[i]["wx"]],
                            [data.oxts[i]["wy"]],
                            [data.oxts[i]["wz"]]])

            # Calculate time difference
            t_now = data.timestamps[i]
            dt = (t_now - t_prev).total_seconds()
            t_prev = t_now

            # MSCKF prediction and measurement update
            msckf.prediction_update(a_m, -w_m, dt)
            msckf.measurement_update(tracks)

            # Show image frame
            # if debug:
            #     cv2.imshow("image", img)
            #     cv2.waitKey(0)

            # Store position
            t.append(data.timestamps[i])
            ground_truth_x.append(dist_E)
            ground_truth_y.append(dist_N)

            p_G = np.dot(rotz(-yaw0), msckf.imu_state.p_G)
            msckf_x.append(p_G[0])
            msckf_y.append(p_G[1])
            # msckf_x.append(msckf.imu_state.p_G[0])
            # msckf_y.append(msckf.imu_state.p_G[1])

            # Store velocity
            msckf_vx.append(msckf.imu_state.v_G[0])
            msckf_vy.append(msckf.imu_state.v_G[1])

        # # Plot
        # if debug:
        #     plt.subplot(311)
        #     plt.plot(ground_truth_x, ground_truth_y, color="red")
        #     plt.plot(msckf_x, msckf_y, color="green")
        #     plt.xlabel("East (m)")
        #     plt.ylabel("North (m)")
        #     plt.axis("equal")
        #
        #     plt.subplot(312)
        #     plt.plot(t,
        #              [data.oxts[i]["vf"] for i in range(len(t))],
        #              color="red")
        #     plt.plot(t, msckf_vx, "green")
        #     plt.xlabel("Date time")
        #     plt.ylabel("Forward Velocity (ms^-1)")
        #
        #     plt.subplot(313)
        #     plt.plot(t,
        #              [data.oxts[i]["vl"] for i in range(len(t))],
        #              color="red")
        #     plt.plot(t, msckf_vy, "green")
        #     plt.xlabel("Date time")
        #     plt.ylabel("Left Velocity (ms^-1)")
        #
        #     plt.show()
