import numpy as np
from numpy import zeros
from numpy import eye as I
from numpy import dot
from numpy import diag
from numpy.linalg import inv
from numpy.matlib import repmat

from prototype.utils.linalg import skew
from prototype.utils.linalg import nullspace
from prototype.utils.quaternion.jpl import quatlcomp
from prototype.utils.quaternion.jpl import quat2euler
from prototype.utils.quaternion.jpl import C
from prototype.viz.plot_matrix import PlotMatrix
from prototype.estimation.msckf.imu_state import IMUState
from prototype.estimation.msckf.camera_state import CameraState
from prototype.estimation.msckf.feature_estimator import FeatureEstimator


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
        # Covariance matrices
        self.P_cam = None
        self.P_imu_cam = None

        # IMU settings
        # -- IMU noise vector
        n_g = kwargs["n_g"].reshape((3, 1))    # Gyro Noise
        n_a = kwargs["n_a"].reshape((3, 1))    # Accel Noise
        n_wg = kwargs["n_wg"].reshape((3, 1))  # Gyro Random Walk Noise
        n_wa = kwargs["n_wa"].reshape((3, 1))  # Accel Random Walk Noise
        n_imu = np.block([[n_g], [n_wg], [n_a], [n_wa]])
        # -- IMU state vector
        imu_q_IG = kwargs.get("imu_q_IG", np.array([0.0, 0.0, 0.0, 1.0]))
        imu_b_g = kwargs.get("imu_b_g", np.zeros((3, 1)))
        imu_v_G = kwargs.get("imu_v_G", np.zeros((3, 1)))
        imu_b_a = kwargs.get("imu_b_a", np.zeros((3, 1)))
        imu_p_G = kwargs.get("imu_p_G", np.zeros((3, 1)))
        self.imu_state = IMUState(imu_q_IG,
                                  imu_b_g,
                                  imu_v_G,
                                  imu_b_a,
                                  imu_p_G,
                                  n_imu)

        # Camera settings
        # -- Camera intrinsics
        self.cam_model = kwargs["cam_model"]
        # -- Camera extrinsics
        self.ext_p_IC = kwargs["ext_p_IC"].reshape((3, 1))
        self.ext_q_CI = kwargs["ext_q_CI"].reshape((4, 1))
        # -- Camera noise
        self.n_u = 1.0e-4
        self.n_v = 1.0e-4
        # -- Camera states
        self.counter_frame_id = 0
        self.cam_states = []
        self.augment_state()

        # Filter settings
        self.enable_ns_trick = kwargs.get("enable_ns_trick", True)
        self.enable_qr_trick = kwargs.get("enable_qr_trick", True)

        # Feature track estimator and settings
        self.feature_estimator = kwargs.get("feature_estimator",
                                            FeatureEstimator())
        self.min_track_length = kwargs.get("min_track_length", 8)

    def augment_state(self):
        """Augment state

        Augment state and covariance matrix with a copy of the current camera
        pose estimate when a new image is recorded

        """
        x_imu_size = self.imu_state.size
        x_cam_size = self.cam_states[0].size if self.N() else 0

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
        imu_q_IG = self.imu_state.q_IG
        imu_p_G = self.imu_state.p_G
        cam_q_CG = dot(quatlcomp(self.ext_q_CI), imu_q_IG)
        cam_p_G = imu_p_G + dot(C(imu_q_IG).T, self.ext_p_IC)
        # -- Add camera state to sliding window
        cam_state = CameraState(self.counter_frame_id, cam_q_CG, cam_p_G)
        self.cam_states.append(cam_state)
        self.counter_frame_id += 1

    def track_cam_states(self, track):
        """Return camera states where feature track was observed

        Parameters
        ----------
        track : FeatureTrack
            Feature track observed

        Returns
        ----------
        track_cam_states : list of CameraState
            M Camera states where feature track of length M was observed

        """
        frame_start = track.frame_start
        frame_end = track.frame_end
        index_start = frame_start - self.cam_states[0].frame_id
        index_end = self.N() - (self.cam_states[-1].frame_id - frame_end)

        track_cam_states = self.cam_states[index_start:index_end]
        assert track_cam_states[0].frame_id == track.frame_start
        assert track_cam_states[-1].frame_id == track.frame_end

        return track_cam_states

    def P(self):
        """Return covariance matrix"""
        if self.N():
            P = np.block([[self.imu_state.P, self.P_imu_cam],
                          [self.P_imu_cam.T, self.P_cam]])
        else:
            P = self.imu_state.P

        return P

    def N(self):
        """Return number of camera states"""
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

        N = self.N()    # Number of camera states
        M = track.tracked_length()  # Length of feature track

        # Measurement jacobian w.r.t feature
        H_f_j = zeros((2 * M, 3))

        # Measurement jacobian w.r.t state
        H_x_j = zeros((2 * M, x_imu_size + x_cam_size * N))

        # Pose index
        pose_idx = track.frame_start - self.cam_states[0].frame_id
        assert track_cam_states[0].frame_id == track.frame_start
        assert track_cam_states[-1].frame_id == track.frame_end
        assert pose_idx == track.frame_start
        assert (pose_idx + (M - 1)) == track.frame_end

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

    def track_residuals(self, cam_model, track, track_cam_states, p_f_G):
        """Calculate track residual

        Parameters
        ----------
        cam_model : CameraModel
            Camera model
        track : FeatureTrack
            Feature track
        track_cam_states : list of CameraState
            Camera states where feature track was observed
        p_G_f : np.array - 3x1
            Estimated feature position in global frame

        Returns
        -------
        r : np.array - 2Nx1
            Residual over camera states where feature was tracked, where N is
            the length of the feature track

        """
        N = len(track_cam_states)
        r_j = np.zeros((2 * N, 1))

        for i in range(N):
            # Transform feature from global frame to i-th camera frame
            C_CG = C(track_cam_states[i].q_CG)
            p_C_f = dot(C_CG, (p_f_G - track_cam_states[i].p_G))
            cu = p_C_f[0, 0] / p_C_f[2, 0]
            cv = p_C_f[1, 0] / p_C_f[2, 0]
            z_hat = np.array([[cu], [cv]])

            # Transform idealized measurement
            z = cam_model.pixel2image(track.track[i].pt).reshape((2, 1))

            # Calculate reprojection error and add it to the residual vector
            rs = 2 * i
            re = 2 * i + 2
            r_j[rs:re, 0] = (z - z_hat).ravel()

        return r_j

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

        # Get M camera states where feature track was tracked
        track_cam_states = self.track_cam_states(track)

        # Estimate j-th feature position in global frame
        p_G_f = self.feature_estimator.estimate(self.cam_model,
                                                track,
                                                track_cam_states)
        if p_G_f is None:
            return (None, None, None)

        # Calculate residuals
        r_j = self.track_residuals(self.cam_model,
                                   track,
                                   track_cam_states,
                                   p_G_f)

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

        # Calculate Kalman gain
        P = self.P()
        K = dot(dot(P, T_H.T), inv(dot(dot(T_H, P), T_H.T) + R_n))

        # Correct states
        dx = dot(K, r_n)
        self.correct_imu_state(dx)
        self.correct_cam_states(dx)

        # Correct covariance matrices
        x_imu_size = self.imu_state.size
        x_cam_size = self.cam_states[0].size
        N = self.N()

        A = I(x_imu_size + x_cam_size * N) - dot(K, T_H)
        P_corrected = dot(A, dot(P, A.T)) + dot(K, dot(R_n, K.T))

        self.imu_state.P = P_corrected[0:x_imu_size, 0:x_imu_size]
        self.P_cam = P_corrected[x_imu_size:, x_imu_size:]
        self.P_imu_cam = P_corrected[0:x_imu_size, x_imu_size:]
