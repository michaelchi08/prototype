import unittest

import matplotlib.pylab as plt

import numpy as np
from numpy import eye as I
from numpy import dot
from numpy import allclose
from numpy import array_equal as np_equal

# import prototype.tests as test
from prototype.utils.linalg import skew
from prototype.utils.linalg import skewsq
from prototype.utils.quaternion.jpl import quat2rot as C
from prototype.utils.quaternion.jpl import euler2quat
from prototype.utils.quaternion.jpl import quat2euler
from prototype.data.kitti import RawSequence
# from prototype.viz.plot_matrix import PlotMatrix
# from prototype.viz.plot_error import plot_error_ellipse

from prototype.msckf.imu_state import IMUState


# GLOBAL VARIABLE
RAW_DATASET = "/data/kitti/raw"


class IMUStateTest(unittest.TestCase):
    def setUp(self):
        self.debug = False
        # self.debug = True

        n_g = np.ones((3, 1)) * 0.01  # Gyro Noise
        n_a = np.ones((3, 1)) * 0.01  # Accel Noise
        n_wg = np.ones((3, 1)) * 0.01  # Gyro Random Walk Noise
        n_wa = np.ones((3, 1)) * 0.01  # Accel Random Walk Noise
        n_imu = np.block([[n_g], [n_wg], [n_a], [n_wa]])

        self.imu_state = IMUState(
            np.array([0.0, 0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            n_imu
        )

    def plot_position(self, pos_true, pos_est):
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
        n_g = np.ones((3, 1)) * 0.01  # Gyro Noise
        n_a = np.ones((3, 1)) * 0.02  # Accel Noise
        n_wg = np.ones((3, 1)) * 0.03  # Gyro Random Walk Noise
        n_wa = np.ones((3, 1)) * 0.04  # Accel Random Walk Noise
        n_imu = np.block([[n_g], [n_wg], [n_a], [n_wa]])

        q_IG = np.array([1.0, 2.0, 3.0, 4.0])
        b_g = np.array([5.0, 6.0, 7.0])
        v_G = np.array([8.0, 9.0, 10.0])
        b_a = np.array([11.0, 12.0, 13.0])
        p_G = np.array([14.0, 15.0, 16.0])

        imu_state = IMUState(q_IG, b_g, v_G, b_a, p_G, n_imu)

        self.assertTrue(np_equal(q_IG, imu_state.q_IG.ravel()))
        self.assertTrue(np_equal(b_g, imu_state.b_g.ravel()))
        self.assertTrue(np_equal(v_G, imu_state.v_G.ravel()))
        self.assertTrue(np_equal(b_a, imu_state.b_a.ravel()))
        self.assertTrue(np_equal(p_G, imu_state.p_G.ravel()))

        self.assertEqual(imu_state.q_IG.shape, (4, 1))
        self.assertEqual(imu_state.b_g.shape, (3, 1))
        self.assertEqual(imu_state.v_G.shape, (3, 1))
        self.assertEqual(imu_state.b_a.shape, (3, 1))
        self.assertEqual(imu_state.p_G.shape, (3, 1))

    def test_F(self):
        w_hat = np.array([1.0, 2.0, 3.0])
        q_hat = np.array([0.0, 0.0, 0.0, 1.0])
        a_hat = np.array([1.0, 2.0, 3.0])
        w_G = np.array([0.1, 0.1, 0.1])

        F = self.imu_state.F(w_hat, q_hat, a_hat, w_G)

        # -- First row --
        self.assertTrue(np_equal(F[0:3, 0:3], -skew(w_hat)))
        self.assertTrue(np_equal(F[0:3, 3:6], -np.eye(3)))
        # -- Third Row --
        self.assertTrue(np_equal(F[6:9, 0:3], dot(-C(q_hat).T, skew(a_hat))))
        self.assertTrue(np_equal(F[6:9, 6:9], -2.0 * skew(w_G)))
        self.assertTrue(np_equal(F[6:9, 9:12], -C(q_hat).T))
        self.assertTrue(np_equal(F[6:9, 12:15], -skewsq(w_G)))
        # -- Fifth Row --
        # self.assertTrue(np_equal(F[12:15, 6:9], np.eye(3)))

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
        self.assertTrue(np_equal(G[0:3, 0:3], -np.eye(3)))
        # -- Second row --
        self.assertTrue(np_equal(G[3:6, 3:6], np.eye(3)))
        # -- Third row --
        self.assertTrue(np_equal(G[6:9, 6:9], -C(q_hat).T))
        # -- Fourth row --
        self.assertTrue(np_equal(G[9:12, 9:12], np.eye(3)))

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
        self.assertTrue(np_equal(J[0:3, 0:3], C_CI))
        # -- Second row --
        self.assertTrue(np_equal(J[3:6, 0:3], skew(dot(C_IG.T, cam_p_IC))))
        # -- Third row --
        self.assertTrue(np_equal(J[3:6, 12:15], I(3)))

        # Plot matrix
        # debug = True
        debug = False
        if debug:
            ax = plt.subplot(111)
            ax.matshow(J)
            plt.show()

    def test_update(self):
        # Setup dataset
        data = RawSequence(RAW_DATASET, "2011_09_26", "0005")

        v0 = data.get_inertial_velocity(0)
        q0 = euler2quat(data.get_attitude(0))

        # Setup IMU state
        n_g = np.ones((3, 1)) * 0.01  # Gyro Noise
        n_a = np.ones((3, 1)) * 0.02  # Accel Noise
        n_wg = np.ones((3, 1)) * 0.03  # Gyro Random Walk Noise
        n_wa = np.ones((3, 1)) * 0.04  # Accel Random Walk Noise
        n_imu = np.block([[n_g], [n_wg], [n_a], [n_wa]])

        q_IG = q0
        b_g = np.zeros((3, 1))
        v_G = v0
        b_a = np.zeros((3, 1))
        p_G = np.zeros((3, 1))

        imu_state = IMUState(q_IG, b_g, v_G, b_a, p_G, n_imu)

        # Setup state history storage and covariance plot
        pos_est = imu_state.p_G
        vel_est = imu_state.v_G
        att_est = quat2euler(imu_state.q_IG)

        # labels = ["theta_x", "theta_y", "theta_z",
        #           "bx_g", "by_g", "bz_g",
        #           "vx", "vy", "vz",
        #           "bx_a", "by_a", "bz_a",
        #           "px", "py", "pz"]
        # P_plot = PlotMatrix(imu_state.P,
        #                     labels=labels,
        #                     show_ticks=True,
        #                     # show=True)
        #                     show=False)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # plt.show(block=False)

        # Loop through data
        for i in range(1, len(data.oxts)):
            a_m, w_m = data.get_imu_measurements(i)
            dt = data.get_dt(i)
            imu_state.update(a_m, w_m, dt)

            # P_plot.update(imu_state.P)

            # plot_error_ellipse(imu_state.p_G[0:2])
            # mean = imu_state.p_G[0:2].ravel()
            # cov = imu_state.P[12:14, 12:14]
            # ax.clear()
            # plot_error_ellipse(mean, cov, ax)
            # fig.canvas.draw()

            pos = imu_state.p_G
            vel = imu_state.v_G
            att = quat2euler(imu_state.q_IG)

            pos_est = np.hstack((pos_est, pos))
            vel_est = np.hstack((vel_est, vel))
            att_est = np.hstack((att_est, att))

        # Plot
        # debug = True
        debug = False
        if debug:
            self.plot_position(data.get_local_position(), pos_est)

            self.plot_velocity(data.get_timestamps(),
                               data.get_inertial_velocity(),
                               vel_est)

            self.plot_attitude(data.get_timestamps(),
                               data.get_attitude(),
                               att_est)
            plt.show()

    def test_correct(self):
        dtheta_CG = np.array([[0.1], [0.2], [0.3]])
        db_g = np.array([[1.1], [1.2], [1.3]])
        dv_G = np.array([[2.1], [2.2], [2.3]])
        db_a = np.array([[3.1], [3.2], [3.3]])
        dp_G = np.array([[4.1], [4.2], [4.3]])
        dx = np.block([[dtheta_CG], [db_g], [dv_G], [db_a], [dp_G]])

        self.imu_state.correct(dx)

        # print(cam)
        expected_q_CG = np.array([0.05, 0.1, 0.15, 0.98])
        self.assertTrue(allclose(expected_q_CG, self.imu_state.q_IG.ravel(),
                                 rtol=1e-2))
        self.assertTrue(allclose(db_g, self.imu_state.b_g))
        self.assertTrue(allclose(dv_G, self.imu_state.v_G))
        self.assertTrue(allclose(db_a, self.imu_state.b_a))
        self.assertTrue(allclose(dp_G, self.imu_state.p_G))

    def test_str(self):
        # Setup IMU state
        n_g = np.ones((3, 1)) * 0.01  # Gyro Noise
        n_a = np.ones((3, 1)) * 0.02  # Accel Noise
        n_wg = np.ones((3, 1)) * 0.03  # Gyro Random Walk Noise
        n_wa = np.ones((3, 1)) * 0.04  # Accel Random Walk Noise
        n_imu = np.block([[n_g], [n_wg], [n_a], [n_wa]])

        q_IG = np.array([0.0, 0.0, 0.0, 1.0])
        b_g = np.zeros((3, 1))
        v_G = np.zeros((3, 1))
        b_a = np.zeros((3, 1))
        p_G = np.zeros((3, 1))

        imu_state = IMUState(q_IG, b_g, v_G, b_a, p_G, n_imu)

        # Test
        imu_state_str = str(imu_state).replace("\t", "")

        # Assert
        expected = """
IMU state:
q:    [ 0.  0.  0.  1.]
b_g:    [ 0.  0.  0.]
p:    [ 0.  0.  0.]
b_a:    [ 0.  0.  0.]
p_G:    [ 0.  0.  0.]
        """.replace("    ", "")
        self.assertEqual(imu_state_str.strip(), expected.strip())
