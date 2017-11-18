from math import sqrt

import numpy as np
from numpy import ones
from numpy import zeros
from numpy import eye as I
from numpy import dot
from numpy import diag
from numpy.linalg import inv
from numpy.matlib import repmat

from prototype.utils.transform import T_camera_global
from prototype.utils.utils import rotnormalize
from prototype.utils.linalg import skew
from prototype.utils.linalg import skewsq
from prototype.utils.linalg import nullspace
from prototype.utils.linalg import enforce_psd
from prototype.utils.quaternion.jpl import quatnormalize
from prototype.utils.quaternion.jpl import quatlcomp
from prototype.utils.quaternion.jpl import quatrcomp
from prototype.utils.quaternion.jpl import quat2euler
from prototype.utils.quaternion.jpl import Omega
from prototype.viz.plot_matrix import PlotMatrix


def C(q):
    q = q.reshape((4, 1))
    R = dot(quatrcomp(q).T, quatlcomp(q))
    R = rotnormalize(R[0:3, 0:3])
    return R


class IMUState:
    """IMU state

    Parameters
    ----------
    q_IG : np.array - 4x1
        JPL Quaternion of IMU in Global frame
    b_g : np.array - 3x1
        Bias of gyroscope
    v_G : np.array - 3x1
        Velocity of IMU in Global frame
    b_a : np.array - 3x1
        Bias of accelerometer
    p_G : np.array - 3x1
        Position of IMU in Global frame

    Attributes
    ----------
    q_IG : np.array - 4x1
        JPL Quaternion of IMU in Global frame
    b_g : np.array - 3x1
        Bias of gyroscope
    v_G : np.array - 3x1
        Velocity of IMU in Global frame
    b_a : np.array - 3x1
        Bias of accelerometer
    p_G : np.array - 3x1
        Position of IMU in Global frame
    w_G : np.array - 3x1
        Gravitational angular velocity
    G_g : np.array - 3x1
        Gravitational acceleration

    """

    def __init__(self, q_IG, b_g, v_G, b_a, p_G, n_imu):
        self.size = 15  # Number of elements in state vector

        # State vector
        self.q_IG = np.array(q_IG).reshape((4, 1))
        self.b_g = np.array(b_g).reshape((3, 1))
        self.v_G = np.array(v_G).reshape((3, 1))
        self.b_a = np.array(b_g).reshape((3, 1))
        self.p_G = np.array(p_G).reshape((3, 1))

        # Constants
        self.w_G = np.array([[0.0], [0.0], [0.0]])
        self.G_g = np.array([[0.0], [0.0], [-9.81]])

        # Covariance matrix
        self.Q = I(12) * n_imu
        self.P = I(self.size)

    def F(self, w_hat, q_hat, a_hat, w_G):
        """Transition Jacobian F matrix

        Predicition or transition matrix in an EKF

        Parameters
        ----------
        w_hat : np.array
            Estimated angular velocity
        q_hat : np.array
            Estimated quaternion (x, y, z, w)
        a_hat : np.array
            Estimated acceleration
        w_G : np.array
            Earth's angular velocity (i.e. Earth's rotation)

        Returns
        -------
        F : np.array - 15x15
            Transition jacobian matrix F

        """
        # F matrix
        F = zeros((15, 15))
        # -- First row --
        F[0:3, 0:3] = -skew(w_hat)
        F[0:3, 3:6] = -ones((3, 3))
        # -- Third Row --
        F[6:9, 0:3] = dot(-C(q_hat).T, skew(a_hat))
        F[6:9, 6:9] = -2.0 * skew(w_G)
        F[6:9, 9:12] = -C(q_hat).T
        F[6:9, 12:15] = -skewsq(w_G)
        # -- Fifth Row --
        F[12:15, 6:9] = ones((3, 3))

        return F

    def G(self, q_hat):
        """Input Jacobian G matrix

        A matrix that maps the input vector (IMU gaussian noise) to the state
        vector (IMU error state vector), it tells us how the inputs affect the
        state vector.

        Parameters
        ----------
        q_hat : np.array
            Estimated quaternion (x, y, z, w)

        Returns
        -------
        G : np.array - 15x12
            Input jacobian matrix G

        """
        # G matrix
        G = zeros((15, 12))
        # -- First row --
        G[0:3, 0:3] = -ones((3, 3))
        # -- Second row --
        G[3:6, 3:6] = ones((3, 3))
        # -- Third row --
        G[6:9, 6:9] = -C(q_hat).T
        # -- Fourth row --
        G[9:12, 9:12] = ones((3, 3))

        return G

    def J(self, cam_q_CI, cam_p_IC, q_hat_IG, N):
        """Jacobian J matrix

        Parameters
        ----------
        cam_q_CI : np.array
            Rotation from IMU to camera frame
            in quaternion (x, y, z, w)
        cam_p_IC : np.array
            Position of camera in IMU frame
        q_hat_IG : np.array
            Rotation from global to IMU frame
        N : float
            Number of camera states

        Returns
        -------
        J: np.array
            Jacobian matrix J

        """
        C_CI = C(cam_q_CI)
        C_IG = C(q_hat_IG)
        J = zeros((6, 15 + 6 * N))

        # -- First row --
        J[0:3, 0:3] = C_CI
        # -- Second row --
        J[3:6, 0:3] = skew(dot(C_IG.T, cam_p_IC))
        J[3:6, 12:15] = I(3)

        return J

    def update(self, a_m, w_m, dt):
        """IMU state update

        Parameters
        ----------
        a_m : np.array
            Accelerometer measurement
        w_m : np.array
            Gyroscope measurement
        dt : float
            Time difference (s)

        Returns
        -------
        a: np.array
            Corrected accelerometer measurement
        w: np.array
            Corrected gyroscope measurement

        """
        # Calculate new accel and gyro estimates
        a_hat = (a_m - self.b_a) * dt
        w_hat = (w_m - self.b_g - dot(C(self.q_IG), self.w_G)) * dt

        # Propagate IMU states
        # -- Orientation
        q_kp1_IG = self.q_IG + dot(0.5 * Omega(w_hat), self.q_IG)  # noqa
        q_kp1_IG = quatnormalize(q_kp1_IG)
        # -- Gyro bias
        b_kp1_g = self.b_g + zeros((3, 1))
        # -- Velocity
        v_kp1_G = self.v_G + dot(C(self.q_IG).T, a_hat) - 2 * dot(skew(self.w_G), self.v_G) * dt - dot(skewsq(self.w_G), self.p_G) * dt + self.G_g * dt  # noqa
        # -- Accel bias
        b_kp1_a = self.b_a + zeros((3, 1))
        # -- Position
        p_kp1_G = self.p_G + self.v_G * dt

        # Update states
        self.q_IG = q_kp1_IG
        self.b_g = b_kp1_g
        self.v_G = v_kp1_G
        self.b_a = b_kp1_a
        self.p_G = p_kp1_G

        # Build the jacobians F and G
        F = self.F(w_hat, self.q_IG, a_hat, self.w_G)
        G = self.G(self.q_IG)

        # Update covariance
        Phi = I(self.size) + F * dt
        self.P = dot(Phi, dot(self.P, Phi.T)) + dot(G, dot(self.Q, G.T)) * dt
        self.P = enforce_psd(self.P)

        return self.P, Phi

    def correct(self, dx):
        """Correct the IMU State

        Parameters
        ----------
        dx : np.array - 6x1
            IMU state correction, where
            dtheta_IG = dx[0:3]
            db_g = dx[3:6]
            dv_G = dx[6:9]
            db_a = dx[9:12]
            dp_G = dx[12:15]

        """
        # Split dx into its own components
        dtheta_IG = dx[0:3].reshape((3, 1))
        db_g = dx[3:6].reshape((3, 1))
        dv_G = dx[6:9].reshape((3, 1))
        db_a = dx[9:12].reshape((3, 1))
        dp_G = dx[12:15].reshape((3, 1))

        # Time derivative of quaternion (small angle approx)
        dq_IG = 0.5 * dtheta_IG
        norm = dot(dq_IG.T, dq_IG)
        if norm > 1.0:
            dq_IG = np.block([[dq_IG], [1.0]]) / sqrt(1.0 + norm)
        else:
            dq_IG = np.block([[dq_IG], [sqrt(1.0 - norm)]])
        dq_IG = quatnormalize(dq_IG)

        # Correct IMU state
        self.q_IG = dot(quatlcomp(dq_IG), self.q_IG)
        self.b_g = self.b_g + db_g
        self.v_G = self.v_G + dv_G
        self.b_a = self.b_a + db_a
        self.p_G = self.p_G + dp_G


class CameraState:
    """Camera state

    Parameters
    ----------
    p_G : np.array
        Position of camera in Global frame
    q_CG : np.array
        Orientation of camera in Global frame

    Attributes
    ----------
    p_G : np.array
        Position of camera in Global frame
    q_CG : np.array
        Orientation of camera in Global frame
    tracks : :obj`list` of :obj`FeatureTrack`
        Feature tracks

    """

    def __init__(self, q_CG, p_G):
        self.size = 6  # Number of elements in state vector

        # State vector
        self.q_CG = np.array(q_CG).reshape((4, 1))
        self.p_G = np.array(p_G).reshape((3, 1))

    def correct(self, dx):
        """Correct the camera state

        Parameters
        ----------
        dx : np.array - 6x1
            Camera state correction, where
            dtheta_IG = dx[0:3]
            dp_G = dx[3:6]

        """
        # Split dx into its own components
        dtheta_CG = dx[0:3].reshape((3, 1))
        dp_G = dx[3:6].reshape((3, 1))

        # Time derivative of quaternion (small angle approx)
        dq_CG = 0.5 * dtheta_CG
        norm = dot(dq_CG.T, dq_CG)
        if norm > 1.0:
            dq_CG = np.block([[dq_CG], [1.0]]) / sqrt(1.0 + norm)
        else:
            dq_CG = np.block([[dq_CG], [sqrt(1.0 - norm)]])
        dq_CG = quatnormalize(dq_CG)

        # Correct camera state
        self.q_CG = dot(quatlcomp(dq_CG), self.q_CG)
        self.p_G = self.p_G + dp_G


class MSCKF:
    """Multi-State Constraint Kalman Filter

    This class implements the MSCKF based on:

        Mourikis, Anastasios I., and Stergios I. Roumeliotis. "A multi-state
        constraint Kalman filter for vision-aided inertial navigation." Robotics
        and automation, 2007 IEEE international conference on. IEEE, 2007.
        APA

        A.I. Mourikis, S.I. Roumeliotis: "A Multi-state Kalman Filter for
        Vision-Aided Inertial Navigation," Technical Report, September 2006
        [http://www.ee.ucr.edu/~mourikis/tech_reports/TR_MSCKF.pdf]

    Attributes
    ----------
    imu_state : IMUState
        IMU State
    n_imu : np.array
        IMU noise
    cam_model : CameraModel
        Camera model
    ext_p_IC : np.array
        Camera extrinsics - position
    ext_q_CI : np.array
        Camera extrinsics - rotation
    P_imu : np.array
        IMU covariance matrix
    P_cam : np.array
        Camera covariance matrix
    P_imu_cam : np.array
        IMU camera covariance matrix

    Parameters
    ----------
    n_g : np.array
        Gryoscope noise
    n_a : np.array
        Accelerometer noise
    n_wg : np.array
        Gryoscope random walk noise
    n_wa : np.array
        Accelerometer random walk noise

    imu_q_IG : np.array - 4x1
        Initial IMU quaternion
    imu_b_g : np.array - 3x1
        Initial gyroscope bias
    imu_v_G : np.array - 3x1
        Initial IMU velocity
    imu_b_a : np.array - 3x1
        Initial accelerometer bias
    imu_p_G : np.array - 3x1
        Initial position

    cam_model : CameraModel
        Camera Model

    enable_ns_trick : bool
        Enable null space trick (default: True)
    enable_qr_trick : bool
        Enable QR decomposition trick (default: True)

    record : bool
        Record filter states (default: False)

    plot_covar : bool
        Plot covariance matrices (default: False)

    """

    def __init__(self, **kwargs):
        # IMU settings
        # -- IMU noise vector
        n_g = kwargs["n_g"]    # Gyro Noise
        n_a = kwargs["n_a"]    # Accel Noise
        n_wg = kwargs["n_wg"]  # Gyro Random Walk Noise
        n_wa = kwargs["n_wa"]  # Accel Random Walk Noise
        n_imu = np.block([n_g.ravel(),
                          n_wg.ravel(),
                          n_a.ravel(),
                          n_wa.ravel()]).reshape((12, 1))
        # -- IMU state vector
        self.imu_state = IMUState(
            kwargs.get("imu_q_IG", np.array([0.0, 0.0, 0.0, 1.0])),
            kwargs.get("imu_b_g", np.array([0.0, 0.0, 0.0])),
            kwargs.get("imu_v_G", np.array([0.0, 0.0, 0.0])),
            kwargs.get("imu_b_a", np.array([0.0, 0.0, 0.0])),
            kwargs.get("imu_p_G", np.array([0.0, 0.0, 0.0])),
            n_imu
        )

        # Camera settings
        # -- Camera noise
        self.n_u = 0.1
        self.n_v = 0.1
        # -- Camera states
        self.cam_states = [CameraState(np.array([0.0, 0.0, 0.0, 1.0]),
                                       np.array([0.0, 0.0, 0.0]))]
        # -- Camera intrinsics
        self.cam_model = kwargs["cam_model"]
        # -- Camera extrinsics
        self.ext_p_IC = kwargs["ext_p_IC"].reshape((3, 1))
        self.ext_q_CI = kwargs["ext_q_CI"].reshape((4, 1))

        # Filter
        self.enable_ns_trick = kwargs.get("enable_ns_trick", True)
        self.enable_qr_trick = kwargs.get("enable_qr_trick", True)

        # Feature track settings
        self.min_track_length = 8

        # Covariance matrices
        x_imu_size = self.imu_state.size      # Size of imu state
        x_cam_size = self.cam_states[0].size  # Size of cam state
        self.P_cam = I(x_cam_size)
        self.P_imu_cam = zeros((x_imu_size, x_cam_size))

        # History
        self.record = kwargs.get("record", True)
        self.pos_est = self.imu_state.p_G
        self.vel_est = self.imu_state.v_G
        self.att_est = quat2euler(self.imu_state.q_IG)

        # Plots
        self.labels_cov = ["theta_x", "theta_y", "theta_z",
                           "bx_g", "by_g", "bz_g",
                           "vx", "vy", "vz",
                           "bx_a", "by_a", "bz_a",
                           "px", "py", "pz"]
        self.P_plot = None
        self.P_imu_plot = None
        self.P_cam_plot = None
        self.P_imu_cam_plot = None
        if kwargs.get("plot_covar", False):
            self.P_plot = PlotMatrix(self.P(), show=True)
            # self.P_imu_plot = PlotMatrix(
            #     self.imu_state.P,
            #     labels=self.labels_cov,
            #     show=True,
            #     show_ticks=True,
            #     show_values=True
            # )
            # self.P_cam_plot = PlotMatrix(
            #     self.P_cam,
            #     show=True
            # )
            # self.P_imu_cam_plot = PlotMatrix(
            #     self.P_imu_cam,
            #     show=True
            # )

    def record_state(self, w_m, dt):
        # Position
        pos = self.imu_state.p_G

        # Velocity
        vel = self.imu_state.v_G

        # Attitude
        att = quat2euler(self.imu_state.q_IG)

        # Store history
        self.pos_est = np.hstack((self.pos_est, pos))
        self.vel_est = np.hstack((self.vel_est, vel))
        self.att_est = np.hstack((self.att_est, att))

    def update_plots(self):
        """Update plots"""
        if self.P_plot:
            self.P_plot.update(self.P())
        if self.P_imu_plot:
            self.P_imu_plot.update(self.imu_state.P)
        if self.P_cam_plot:
            self.P_cam_plot.update(self.P_cam)
        if self.P_imu_cam_plot:
            self.P_imu_cam_plot.update(self.P_imu_cam)

    def P(self):
        """Covariance matrix"""
        P = np.block([[self.imu_state.P, self.P_imu_cam],
                      [self.P_imu_cam.T, self.P_cam]])

        return P

    def N(self):
        """Number of camera states"""
        return len(self.cam_states)

    def H(self, track, track_cam_states, p_G_f):
        """Form the Jacobian measurement matrix

        - $H^{(j)}_x$ of j-th feature track with respect to state $X$
        - $H^{(j)}_f$ of j-th feature track with respect to feature position

        Parameters
        ----------
        track : FeatureTrack
            Feature track of length M
        track_cam_states : list of CameraState
            N Camera states
        p_G_f : np.array
            Feature position in global frame

        Returns
        -------
        H_x_j: np.matrix
            Measurement jacobian matrix w.r.t state (size: 2*M x (15+6)N)

        """
        x_imu_size = self.imu_state.size      # Size of imu state
        x_cam_size = self.cam_states[0].size  # Size of cam state

        N = len(self.cam_states)    # Number of camera states
        M = track.tracked_length()  # Length of feature track

        # Measurement jacobian w.r.t feature
        H_f_j = zeros((2 * M, 3))

        # Measurement jacobian w.r.t state
        H_x_j = zeros((2 * M, x_imu_size + x_cam_size * N))

        # Pose index, minus 1 because track is not observed in last cam state
        pose_idx = N - M - 1

        # Form measurement jacobians
        for i in range(M):
            # Feature position in camera frame
            C_CG = C(track_cam_states[i].q_CG)
            p_G_C = track_cam_states[i].p_G
            p_C_f = dot(C_CG, (p_G_f - p_G_C))
            X, Y, Z = p_C_f.ravel()

            # dh / dg
            dhdg = (1.0 / Z) * np.array([[1.0, 0.0, -X / Z],
                                         [0.0, 1.0, -Y / Z]])

            # Row start and end index
            rs = 2 * i
            re = 2 * i + 2

            # Column start and end index
            cs_dhdq = x_imu_size + (x_cam_size * pose_idx)
            ce_dhdq = x_imu_size + (x_cam_size * pose_idx) + 3

            cs_dhdp = x_imu_size + (x_cam_size * pose_idx) + 3
            ce_dhdp = x_imu_size + (x_cam_size * pose_idx) + 6

            # H_f_j measurement jacobian w.r.t feature
            H_f_j[rs:re, :] = dot(dhdg, C_CG)

            # H_x_j measurement jacobian w.r.t state
            H_x_j[rs:re, cs_dhdq:ce_dhdq] = dot(dhdg, skew(p_C_f))
            H_x_j[rs:re, cs_dhdp:ce_dhdp] = dot(-dhdg, C_CG)

            # Update pose_idx
            pose_idx += 1

        return H_f_j, H_x_j

    def prediction_update(self, a_m, w_m, dt):
        """IMU state update

        Parameters
        ----------
        a_m : np.array
            Accelerometer measurement
        w_m : np.array
            Gyroscope measurement
        dt : float
            Time difference (s)

        """
        # Update covariance matrices
        self.P_imu, Phi = self.imu_state.update(a_m, w_m, dt)
        self.P_cam = self.P_cam
        self.P_imu_cam = dot(Phi, self.P_imu_cam)

        if self.record:
            self.record_state(w_m, dt)

    def triangulate(self, pt1, pt2, C_C0C1, t_C0_C1C0):
        """Triangulate feature observed from camera C0 and C1 and return the
        position relative to camera C0

        Parameters
        ----------
        pt1 : np.array - 2x1
            Feature point 1 in pixel coordinates
        pt2 : np.array - 2x1
            Feature point 2 in pixel coordinates
        C_C0C1 : np.array - 3x3
            Rotation matrix from frame C1 to C0
        t_C0_C0C1 : np.array - 3x3
            Translation vector from frame C0 to C1 expressed in C0

        Returns
        -------
        p_C0_f : np.array - 3x1
            Returns feature point expressed in C0 in camera coordinates

        """
        # Convert points to homogenous coordinates and normalize
        pt1 = np.block([[pt1], [1.0]])
        pt2 = np.block([[pt2], [1.0]])
        pt1 = pt1 / np.linalg.norm(pt1)
        pt2 = pt2 / np.linalg.norm(pt2)

        # Triangulate
        A = np.block([pt1, -dot(C_C0C1, pt2)])
        b = t_C0_C1C0
        result = np.linalg.lstsq(A, b)
        x, residual, rank, s = result
        p_C0_f = x[0] * pt1

        return p_C0_f

    def estimate_feature(self, cam_model, track, track_cam_states, debug=False):
        """Estimate feature 3D location by optimizing over inverse depth
        parameterization using Gauss Newton Optimization

        Parameters
        ----------
        cam_model : CameraModel
            Camera model
        track : FeatureTrack
            Feature track
        track_cam_states : list of CameraState
            Camera states where feature track was observed
        debug :
             (Default value = False)

        Returns
        -------
        p_G_f : np.array - 3x1
            Estimated feature position in global frame
        k : int
            Optimized over k iterations
        r : np.array - 2Nx1
            Residual np.array over all camera states

        """
        # Calculate rotation and translation of first and last camera states
        C_C0G = C(track_cam_states[0].q_CG)
        C_C1G = C(track_cam_states[-1].q_CG)
        p_G_C0 = track_cam_states[0].p_G
        p_G_C1 = track_cam_states[-1].p_G
        # -- Set camera 0 as origin, work out rotation and translation of
        # -- camera 1 relative to to camera 0
        C_C0C1 = dot(C_C0G, C_C1G.T)
        t_C0_C1C0 = dot(C_C0G, (p_G_C1 - p_G_C0))
        # -- Convert from pixel coordinates to image coordinates
        pt1 = cam_model.pixel2image(track.track[0].pt).reshape((2, 1))
        pt2 = cam_model.pixel2image(track.track[-1].pt).reshape((2, 1))

        # Calculate initial estimate of 3D position
        X = self.triangulate(pt1, pt2, C_C0C1, t_C0_C1C0)

        # Create inverse depth params (these are to be optimized)
        alpha = X[0, 0] / X[2, 0]
        beta = X[1, 0] / X[2, 0]
        rho = 1.0 / X[2, 0]

        # Gauss Newton optimization
        C_C0G = C(track_cam_states[0].q_CG)
        p_G_C0 = track_cam_states[0].p_G
        max_iter = 30
        r_Jprev = float("inf")  # residual jacobian

        for k in range(max_iter):
            N = len(track_cam_states)
            r = zeros((2 * N, 1))
            J = zeros((2 * N, 3))
            W = zeros((2 * N, 2 * N))

            # Calculate residuals
            for i in range(N):
                # Get camera current rotation and translation
                C_CiG = C(track_cam_states[i].q_CG)
                p_G_Ci = track_cam_states[i].p_G

                # Set camera 0 as origin, work out rotation and translation
                # of camera i relative to to camera 0
                C_Ci_C0 = dot(C_CiG, C_C0G.T)
                t_Ci_CiC0 = dot(C_CiG, (p_G_C0 - p_G_Ci))

                # Project estimated feature location to image plane
                h = dot(C_Ci_C0, np.array([[alpha], [beta], [1]])) + rho * t_Ci_CiC0  # noqa

                # Calculate reprojection error
                # -- Convert measurment to normalized pixel coordinates
                z = cam_model.pixel2image(track.track[i].pt).reshape((2, 1))
                # -- Convert feature location to normalized pixel coordinates
                x = np.array([h[0] / h[2], h[1] / h[2]])
                # -- Reprojcetion error
                r[2 * i:(2 * (i + 1))] = z - x

                # Form the Jacobian
                drdalpha = np.array([
                    -C_CiG[0, 0] / h[2, 0] + (h[0, 0] / h[2, 0]**2) * C_CiG[2, 0],  # noqa
                    -C_CiG[1, 0] / h[2, 0] + (h[1, 0] / h[2, 0]**2) * C_CiG[2, 0]   # noqa
                ])
                drdbeta = np.array([
                    -C_CiG[0, 1] / h[2, 0] + (h[0, 0] / h[2, 0]**2) * C_CiG[2, 1],  # noqa
                    -C_CiG[1, 1] / h[2, 0] + (h[1, 0] / h[2, 0]**2) * C_CiG[2, 1]   # noqa
                ])
                drdrho = np.array([
                    -t_Ci_CiC0[0, 0] / h[2, 0] + (h[0, 0] / h[2, 0]**2) * t_Ci_CiC0[2, 0],  # noqa
                    -t_Ci_CiC0[1, 0] / h[2, 0] + (h[1, 0] / h[2, 0]**2) * t_Ci_CiC0[2, 0]   # noqa
                ])
                J[2 * i:(2 * (i + 1)), 0] = drdalpha.ravel()
                J[2 * i:(2 * (i + 1)), 1] = drdbeta.ravel()
                J[2 * i:(2 * (i + 1)), 2] = drdrho.ravel()

                # Form the weight matrix
                W[2 * i:(2 * (i + 1)), 2 * i:(2 * (i + 1))] = np.diag([0.00001, 0.00001]) # noqa

            # Check hessian if it is numerically ok
            H_approx = dot(J.T, J)
            if np.linalg.cond(H_approx) > 1e3:
                return None, None, None

            # Update estimate params using Gauss Newton
            delta = dot(inv(H_approx), dot(J.T, r))
            theta_k = np.array([[alpha], [beta], [rho]]) + delta
            alpha = theta_k[0, 0]
            beta = theta_k[1, 0]
            rho = theta_k[2, 0]

            # Check how fast the residuals are converging to 0
            r_Jnew = float(0.5 * dot(r.T, r))
            if r_Jnew < 0.0001:
                break
            r_J = abs((r_Jnew - r_Jprev) / r_Jnew)
            r_Jprev = r_Jnew

            # Break loop if not making any progress
            if r_J < 0.000001:
                break

            # # Update params using Weighted Gauss Newton
            # JWJ = dot(J.T, np.linalg.solve(W, J))
            # delta = np.linalg.solve(JWJ, dot(-J.T, np.linalg.solve(W, r)))
            # theta_k = np.array([[alpha], [beta], [rho]]) + delta
            # alpha = theta_k[0, 0]
            # beta = theta_k[1, 0]
            # rho = theta_k[2, 0]

            # # Check how fast the residuals are converging to 0
            # r_Jnew = float(0.5 * dot(r.T, r))
            # if r_Jnew < 0.0000001:
            #     break
            # r_J = abs((r_Jnew - r_Jprev) / r_Jnew)
            # r_Jprev = r_Jnew
            #
            # # Break loop if not making any progress
            # if r_J < 0.000001:
            #     break

        # Debug
        if debug:
            # print(k)
            # print(alpha)
            # print(beta)
            # print(rho)
            print("track_length: ", track.tracked_length())
            print("iterations: ", k)
            print("residual: ", r)

        # Convert estimated inverse depth params back to feature position in
        # global frame.  See (Eq.38, Mourikis2007 (A Multi-State Constraint
        # Kalman Filter for Vision-aided Inertial Navigation)
        z = 1 / rho
        X = np.array([[alpha], [beta], [1.0]])
        C_C0G = C(track_cam_states[0].q_CG)
        p_G_C0 = track_cam_states[0].p_G
        p_G_f = z * dot(C_C0G.T, X) + p_G_C0

        return (p_G_f, k, r)

    def augment_state(self):
        """Augment state

        Augment state and covariance matrix with a copy of the current camera
        pose estimate when a new image is recorded

        """
        x_imu_size = self.imu_state.size
        x_cam_size = self.cam_states[0].size

        # Camera pose Jacobian
        J = self.imu_state.J(self.ext_q_CI, self.ext_p_IC,
                             self.imu_state.q_IG, self.N())

        # Augment MSCKF covariance matrix (with new camera state)
        X = np.block([[I(x_imu_size + x_cam_size * self.N())], [J]])
        P = dot(X, dot(self.P(), X.T))
        self.imu_state.P = P[0:x_imu_size, 0:x_imu_size]
        self.P_cam = P[x_imu_size:, x_imu_size:]
        self.P_imu_cam = P[0:x_imu_size, x_imu_size:]

        # Add new camera state to sliding window by using current IMU pose
        # estimate to calculate camera pose
        # -- Create camera state in global frame
        cam_q_CG = dot(quatlcomp(self.ext_q_CI), self.imu_state.q_IG)
        cam_p_G = self.imu_state.p_G + dot(C(self.imu_state.q_IG).T, self.ext_p_IC)  # noqa
        # -- Transform camera state from global frame to camera frame
        cam_q_CG = T_camera_global * cam_q_CG
        cam_p_G = T_camera_global * cam_p_G
        # -- Add camera state to sliding window
        self.cam_states.append(CameraState(cam_q_CG, cam_p_G))

    def residualize_track(self, track):
        """Residualize feature track

        Parameters
        ----------
        track : FeatureTrack
            Feature track to residualize

        Returns
        -------
        H_o_j: (np.array)
            Measurement jacobian w.r.t state projected to null space
        r_o_j: (np.array)
            Residual vector projected to null space
        R_o_j: (np.array)
            Covariance matrix projected to null space

        """
        # Pre-check
        if track.tracked_length() < self.min_track_length:
            return (None, None, None)

        # Get last M camera states where feature track was tracked
        M = track.tracked_length()
        track_cam_states = self.cam_states[-M - 1: -1]

        # Estimate j-th feature position in global frame
        p_G_f, k, r_j = self.estimate_feature(self.cam_model,
                                              track,
                                              track_cam_states)
        if p_G_f is None:
            return (None, None, None)

        # Form jacobian of measurement w.r.t both state and feature
        H_f_j, H_x_j = self.H(track, track_cam_states, p_G_f)

        # Form the covariance matrix of different feature observations
        nb_residuals = np.size(r_j)
        sigma_img = np.array([self.n_u, self.n_v])
        sigma_img = repmat(sigma_img, 1, int(nb_residuals / 2))
        R_j = diag(sigma_img.ravel())

        # Perform Null Space Trick?
        if self.enable_ns_trick:
            # Perform null space trick to decorrelate feature position error
            # away state errors by removing the measurement jacobian w.r.t.
            # feature position via null space projection [Section D:
            # Measurement Model, Mourikis2007]
            A_j = nullspace(H_f_j.T)
            H_o_j = dot(A_j.T, H_x_j)
            r_o_j = dot(A_j.T, r_j)
            R_o_j = dot(A_j.T, dot(R_j, A_j))

        else:
            H_o_j = H_x_j
            r_o_j = r_j
            R_o_j = R_j

        return H_o_j, r_o_j, R_o_j

    def stack_residuals(self, H_o, r_o, R_o, H_o_j, r_o_j, R_o_j):
        # Initialize H_o, r_o and R_o
        if H_o is None and r_o is None and R_o is None:
            H_o = H_o_j
            r_o = r_o_j
            R_o = R_o_j

        # Stack H_o, r_o and R_o
        else:
            H_o = np.vstack((H_o, H_o_j))
            r_o = np.vstack((r_o, r_o_j))

            # R_o is a bit special, it is the covariance matrix
            # so it has to be stacked diagonally
            R_o = np.block([
                [R_o, zeros((R_o.shape[0], R_o_j.shape[1]))],
                [zeros((R_o_j.shape[0], R_o.shape[1])), R_o_j]
            ])

        return H_o, r_o, R_o

    def calculate_residuals(self, tracks):
        """Calculate residuals

        Parameters
        ----------
        tracks : list of FeatureTrack
            List of feature tracks

        Returns
        -------
        T_H: np.array
            QR decomposition
        r_n: np.array
            Residual vector r
        R_n: np.array
            Covariance matrix R

        """
        H_o = None
        r_o = None
        R_o = None

        # Residualize feature tracks
        for track in tracks:
            H_o_j, r_o_j, R_o_j = self.residualize_track(track)

            # Check if track is bad
            if H_o_j is None and r_o_j is None and R_o_j is None:
                continue  # Bad track, skipping
            else:
                H_o, r_o, R_o = self.stack_residuals(H_o, r_o, R_o,
                                                     H_o_j, r_o_j, R_o_j)

        # No residuals, do not continue
        if H_o is None and r_o is None and R_o is None:
            return (None, None, None)

        # Perform QR decomposition?
        if self.enable_qr_trick:
            # Perform QR decomposition to reduce computation in actual EKF
            # update, since R_o for 10 features seen in 10 camera poses each
            # would yield a R_o of dimension 170 [Section E: EKF Updates,
            # Mourikis2007]
            Q, R = np.linalg.qr(H_o)
            # -- Find non-zero rows
            nonzero_rows = [np.any(row != 0) for row in R]
            # -- Remove rows that are all zeros
            T_H = R[nonzero_rows, :]
            Q_1 = Q[:, nonzero_rows]
            # -- Calculate residual
            r_n = dot(Q_1.T, r_o)
            R_n = dot(Q_1.T, dot(R_o, Q_1))

        else:
            T_H = H_o
            r_n = r_o
            R_n = R_o

        return T_H, r_n, R_n

    def correct_imu_state(self, dx):
        """Correct IMU state

        Parameters
        ----------
        dx : np.array
            State correction vector

        """
        dx_imu = dx[0:self.imu_state.size]
        self.imu_state.correct(dx_imu)

    def correct_cam_states(self, dx):
        """Correct camera states

        Parameters
        ----------
        dx : np.array
            State correction vector

        """
        x_imu_size = self.imu_state.size
        x_cam_size = self.cam_states[0].size

        for i in range(self.N()):
            rs = x_imu_size + x_cam_size * i
            re = x_imu_size + x_cam_size * i + x_cam_size
            dx_cam = dx[rs:re]
            self.cam_states[i].correct(dx_cam)

    def measurement_update(self, tracks):
        """Measurement update

        Parameters
        ----------
        tracks : list of FeatureTrack
            List of feature tracks

        """
        # Add a camera state to state vector
        self.augment_state()

        # Continue with EKF update?
        if len(tracks) == 0:
            return

        # Calculate residuals
        T_H, r_n, R_n = self.calculate_residuals(tracks)
        if T_H is None and r_n is None and R_n is None:
            return

        print(r_n.shape)

        # Calculate Kalman gain
        K = dot(self.P(), dot(T_H.T, inv(dot(T_H, dot(self.P(), T_H.T)) + R_n)))

        # Correct states
        dx = dot(K, r_n)
        self.correct_imu_state(dx)
        self.correct_cam_states(dx)

        # Correct covariance matrices
        x_imu_size = self.imu_state.size
        x_cam_size = self.cam_states[0].size

        A = I(x_imu_size + x_cam_size * self.N()) - dot(K, T_H)
        P_corrected = dot(A, dot(self.P(), A.T)) + dot(K, dot(R_n, K.T))

        self.imu_state.P = P_corrected[0:x_imu_size, 0:x_imu_size]
        self.P_cam = P_corrected[x_imu_size:, x_imu_size:]
        self.P_imu_cam = P_corrected[0:x_imu_size, x_imu_size:]
