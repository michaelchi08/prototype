import numpy as np
from numpy import ones
from numpy import zeros
from numpy import eye as I
from numpy import dot
from numpy.linalg import inv

from prototype.utils.linalg import skew
from prototype.utils.linalg import nullspace
from prototype.utils.quaternion.jpl import quatmul
from prototype.utils.quaternion.jpl import quat2rot as C
from prototype.utils.quaternion.jpl import Omega
from prototype.vision.geometry import triangulate_point


class CameraState:
    """ Camera state """
    def __init__(self, p_G, q_CG):
        """ Constructor

        Args:

            p_G (np.array): Position of camera in Global frame
            q_CG (np.array): Orientation of camera in Global frame

        """
        self.p_G = np.array(p_G).reshape((3, 1))
        self.q_CG = np.array(q_CG).reshape((4, 1))


class IMUState:
    """ IMU state """
    def __init__(self, q_IG, b_g, v_G, b_a, p_G):
        """ Constructor

        Args:

            q_IG (np.array - 4x1): JPL Quaternion of IMU in Global frame
            b_g (np.array - 3x1): Bias of gyroscope
            v_G (np.array - 3x1): Velocity of IMU in Global frame
            b_a (np.array - 3x1): Bias of accelerometer
            p_G (np.array - 3x1): Position of IMU in Global frame

        """
        self.q_IG = np.array(q_IG).reshape((4, 1))
        self.b_g = np.array(b_g).reshape((3, 1))
        self.v_G = np.array(p_G).reshape((3, 1))
        self.b_a = np.array(b_g).reshape((3, 1))
        self.p_G = np.array(p_G).reshape((3, 1))

        self.w_G = np.array([[0.0], [0.0], [0.0]])
        self.G_g = np.array([[0.0], [0.0], [9.81]])

    def update(self, a_m, w_m, dt):
        """ IMU state update """
        # Calculate new accel and gyro estimates
        a = a_m - self.b_a * dt
        w = w_m - self.b_g - dot(C(self.q_IG), self.w_G) * dt

        # Propagate IMU states
        q_kp1_IG = self.q_IG + 0.5 * dot(Omega(w), self.q_IG)  # noqa
        b_kp1_g = self.b_g + zeros((3, 1))
        v_kp1_G = self.v_G + dot(C(self.q_IG), a) - 2 * dot(skew(self.w_G), self.v_G) - dot(skew(self.w_G)**2, self.p_G) + self.G_g  # noqa
        b_kp1_a = self.b_a + zeros((3, 1))
        p_kp1_G = self.v_G

        # Update states
        self.q_IG = q_kp1_IG
        self.b_g = b_kp1_g
        self.v_G = v_kp1_G
        self.b_a = b_kp1_a
        self.p_G = p_kp1_G

        return (a, w)


class MSCKF:
    """ Multi-State Constraint Kalman Filter

    This class implements the MSCKF based on the paper:

        Mourikis, Anastasios I., and Stergios I. Roumeliotis. "A multi-state
        constraint Kalman filter for vision-aided inertial navigation." Robotics
        and automation, 2007 IEEE international conference on. IEEE, 2007.
        APA

    """
    def __init__(self, **kwargs):
        n_g = kwargs["n_g"]    # Gyro Noise
        n_a = kwargs["n_a"]    # Accel Noise
        n_wg = kwargs["n_wg"]  # Gyro Random Walk Noise
        n_wa = kwargs["n_wa"]  # Accel Random Walk Noise

        # IMU state vector
        self.imu_state = IMUState(
            kwargs.get("imu_q_IG", np.array([0.0, 0.0, 0.0, 1.0])),
            kwargs.get("imu_b_g", np.array([0.0, 0.0, 0.0])),
            kwargs.get("imu_v_G", np.array([0.0, 0.0, 0.0])),
            kwargs.get("imu_b_a", np.array([0.0, 0.0, 0.0])),
            kwargs.get("imu_p_G", np.array([0.0, 0.0, 0.0]))
        )

        # IMU noise vector
        self.n_imu = np.block([n_g.ravel(),
                               n_wg.ravel(),
                               n_a.ravel(),
                               n_wa.ravel()]).reshape((12, 1))

        # IMU noise covariance matrix
        self.Q_imu = I(12) * self.n_imu

        # Covariance matrices
        self.P_imu = I(15)
        self.P_cam = I(6)
        self.P_imu_cam = I(15)

        # Camera states
        self.cam_states = []

        # Camera extrinsics
        self.ext_p_IC = np.array([0.0, 0.0, 0.0]).reshape((3, 1))
        self.ext_q_CI = np.array([1.0, 0.0, 0.0, 0.0]).reshape((4, 1))

    def F(self, w_hat, q_hat, a_hat, w_G):
        """ Transition Jacobian F matrix

        Aka predicition or transition matrix in an EKF

        Args:

            w_hat (np.array): Estimated angular velocity
            q_hat (np.array): Estimated quaternion (x, y, z, w)
            a_hat (np.array): Estimated acceleration
            w_G (np.array): Earth's angular velocity (i.e. Earth's rotation)

        Returns:

            Numpy matrix of size 15x15

        """
        # F matrix
        F = zeros((15, 15))
        # -- First row --
        F[0:3, 0:3] = -skew(w_hat)
        F[0:3, 3:6] = ones((3, 3))
        # -- Third Row --
        F[6:9, 0:3] = dot(-C(q_hat).T, skew(a_hat))
        F[6:9, 6:9] = -2.0 * skew(w_G)
        F[6:9, 9:12] = -C(q_hat).T
        F[6:9, 12:15] = -skew(w_G)**2
        # -- Fifth Row --
        F[12:15, 6:9] = ones((3, 3))

        return F

    def G(self, q_hat):
        """ Input Jacobian G matrix

        A matrix that maps the input vector (IMU gaussian noise) to the state
        vector (IMU error state vector), it tells us how the inputs affect the
        state vector.

        Args:

            q_hat (np.array): Estimated quaternion (x, y, z, w)

        Returns:

            Numpy matrix of size 15x12

        """
        # G matrix
        G = zeros((15, 12))
        # -- First row --
        G[0:3, 0:3] = ones((3, 3))
        # -- Second row --
        G[3:6, 3:6] = ones((3, 3))
        # -- Third row --
        G[6:9, 6:9] = -C(q_hat).T
        # -- Fourth row --
        G[9:12, 9:12] = ones((3, 3))

        return G

    def J(self, cam_q_C_I, cam_p_I_C, q_hat_IG, N):
        """ Jacobian J matrix

        Args:

            cam_q_C_I (np.array): Rotation from IMU to camera frame
                                  in quaternion (x, y, z, w)
            cam_p_I_C (np.array): Position of camera in IMU frame
            q_hat_IG (np.array): Rotation from global to IMU frame

        """
        q_hat_IG = self.imu_state.q_IG
        C_C_I = C(cam_q_C_I)
        C_IG = C(q_hat_IG)
        J = zeros(6, 15 + 6 * N)

        # -- First row --
        J[0:3, 0:3] = C_C_I
        # -- Second row --
        J[3:6, 0:3] = skew(dot(C_IG.T, cam_p_I_C))
        J[3:6, 9:12] = I(3)

        return J

    def P(self):
        P = np.block([[self.P_imu, self.P_imu_cam],
                      [self.P_imu_cam.T, self.P_cam]])

        return P

    def H(self, cam_states, track, p_G_f):
        """ Form the Jacobian measurement matrix

        - $H^{(j)}_x$ of j-th feature track with respect to state $X$
        - $H^{(j)}_f$ of j-th feature track with respect to feature position

        Args:

            cam_states (list of CameraState): N Camera states
            track (FeatureTrack): Feature track of length M

        Returns:

            H_x_j jacobian matrix (np.matrix - 2*M x (12+6)N)

        """
        N = len(cam_states)
        M = track.tracked_length()
        H_f_j = zeros(2 * M, 3)
        H_x_j = zeros(2 * M, 12 + 6 * N)

        # Form measurement jacobians
        pose_idx = N - M
        for i in range(M):
            # Feature position in camera frame
            C_C_G = C(cam_states[pose_idx].q_CG)
            p_G_C = cam_states[pose_idx].p_G
            p_C_f = C_C_G * (p_G_f - p_G_C)
            X, Y, Z = p_C_f.ravel()

            # dh / dg
            dhdg = (1.0 / Z) * np.array([[1.0, 0.0, -X / Z],
                                         [0.0, 1.0, -Y / Z]])

            # H_f_j measurement jacobian w.r.t feature
            H_f_j[(2 * i):(2 * i + 2), :] = dhdg * C_C_G

            # H_x_j measurement jacobian w.r.t state
            H_x_j[(2 * i):(2 * i + 2), 12 + 6 * pose_idx + 1:12 + 6 * (pose_idx) + 3] = dhdg * skew(p_C_f)  # noqa
            H_x_j[(2 * i):(2 * i + 2), 12 + 6 * pose_idx + 4:12 + 6 * (pose_idx) + 6] = -dhdg * C_C_G       # noqa

        # Perform null space tricks as per Mourikis 2007
        A_j = nullspace(H_f_j.T)
        H_o_j = A_j.T * H_x_j

        return H_x_j, H_o_j, A_j

    def prediction_update(self, a_m, w_m, dt):
        """ IMU state update """
        # Propagate IMU state
        a_hat, w_hat = self.imu_state.update(a_m, w_m, dt)

        # Build the jacobians F and G
        F = self.F(w_hat, self.imu_state.q_IG, a_hat, self.imu_state.w_G)
        G = self.G(self.imu_state.q_IG)

        # State transition matrix
        Phi = I(15) + F * dt

        # Update covariance matrices
        self.P_imu = dot(Phi, self.P_imu) + dot(self.P_imu, Phi.T) + dot(G, dot(self.Q_imu, G.T))  # noqa
        self.P_cam = self.P_cam
        self.P_imu_cam = dot(Phi, self.P_imu_cam)

    def estimate_feature(self, cam_model, track, track_cam_states, debug=False):
        """ Estimate feature 3D location by optimizing over inverse depth
        paramterization using Gauss Newton Optimization

        Args:

            cam_model (CameraModel): Camera model
            track (FeatureTrack): Feature track
            track_cam_states (list of CameraState): Camera states where feature
                                                    track was observed
            K (np.array): Camera intrinsics

        Returns:

            (p_G_f, k, r)

            p_G_f (np.array - 3x1): Estimated feature position in global frame
            k (int): Optimized over k iterations
            r (np.array - size 2Nx1): Residual np.array over all camera states

        """
        # Calculate initial estimate of 3D position
        # -- Calculate rotation and translation of camera 0 and 1
        C_C0G = C(track_cam_states[0].q_CG)
        C_C1G = C(track_cam_states[1].q_CG)
        p_G_C0 = track_cam_states[0].p_G
        p_G_C1 = track_cam_states[1].p_G
        # -- Set camera 0 as origin, work out rotation and translation of
        # -- camera 1 relative to to camera 0
        C_C0C1 = dot(C_C0G, C_C1G.T)
        t_C0_C1C0 = dot(C_C0G, (p_G_C1 - p_G_C0))
        # -- Triangulate
        x1 = np.block([track.track[0].pt, 1.0])
        x2 = np.block([track.track[1].pt, 1.0])
        P1 = cam_model.P(np.eye(3), ones((3, 1)))
        P2 = cam_model.P(C_C0C1, t_C0_C1C0)
        X = triangulate_point(x1, x2, P1, P2)

        # Create inverse depth params (these are to be optimized)
        alpha = X[0] / X[2]
        beta = X[1] / X[2]
        rho = 1.0 / X[2]

        # Gauss Newton optimization
        r_Jprev = float("inf")  # residual jacobian

        for k in range(10):
            N = len(track_cam_states)
            r = zeros((2 * N, 1))
            J = zeros((2 * N, 3))

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
                h = dot(C_Ci_C0, np.array([[alpha], [beta], [1]])) + dot(rho, t_Ci_CiC0)  # noqa

                # Calculate reprojection error
                # -- Camera intrinsics
                cx, cy = cam_model.K[0, 2], cam_model.K[1, 2]
                fx, fy = cam_model.K[0, 0], cam_model.K[1, 1]
                # -- Convert measurment to normalized pixel coordinates
                z = track.track[i].pt
                z = np.array([[(z[0] - cx) / fx], [(z[1] - cy) / fy]])
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
                    -t_Ci_CiC0[0] / h[2, 0] + (h[0, 0] / h[2, 0]**2) * t_Ci_CiC0[2],  # noqa
                    -t_Ci_CiC0[1] / h[2, 0] + (h[1, 0] / h[2, 0]**2) * t_Ci_CiC0[2]   # noqa
                ])
                J[2 * i:(2 * (i + 1)), 0] = drdalpha.ravel()
                J[2 * i:(2 * (i + 1)), 1] = drdbeta.ravel()
                J[2 * i:(2 * (i + 1)), 2] = drdrho.ravel()

            # Update esitmated params using Gauss Newton
            delta = dot(inv(dot(J.T, J)), dot(J.T, r))
            theta_k = np.array([[alpha], [beta], [rho]]) - delta
            alpha = theta_k[0, 0]
            beta = theta_k[1, 0]
            rho = theta_k[2, 0]

            # Check how fast the residuals are converging to 0
            r_Jnew = float(0.5 * dot(r.T, r))
            if r_Jnew < 0.000001:
                break
            r_J = abs((r_Jnew - r_Jprev) / r_Jnew)
            r_Jprev = r_Jnew

            # Break loop if not making any progress
            if r_J < 0.000001:
                break

        # Debug
        if debug:
            print(k)
            print(alpha)
            print(beta)
            print(rho)

        # Convert estimated inverse depth params back to feature position in
        # global frame.  See (Eq.38, Mourikis2007 (A Multi-State Constraint
        # Kalman Filter for Vision-aided Inertial Navigation)
        z = 1 / rho
        X = np.array([[alpha], [beta], [1.0]])
        C_C0G = C(track_cam_states[0].q_CG)
        p_G_C0 = track_cam_states[0].p_G
        p_G_f = dot(z, dot(C_C0G.T, X)) + p_G_C0

        return (p_G_f, k, r)

    def calculate_track_residual(self,
                                 cam_model,
                                 track,
                                 track_cam_states,
                                 p_G_f,
                                 debug=False):
        """ Calculate the residual of a single feature track

        Args:

            p_G_f (np.array - 3x1): Feature position in global frame
            track (FeatureTrack): A single feature track
            track_cam_states (list of CameraState): N Camera states where
                                                    feature track was observed

        Returns:

            Residual vector (np.array - 2*Nx1)

        """
        # Residual vector
        r_j = []

        # Camera intrinsics
        cx, cy = cam_model.K[0, 2], cam_model.K[1, 2]
        fx, fy = cam_model.K[0, 0], cam_model.K[1, 1]

        # Calculate residual vector
        for i in range(len(track_cam_states)):
            # Transform feature position from global to camera frame at pose i
            C_C_G = C(track_cam_states[i].q_CG)
            p_C_f = dot(C_C_G, (p_G_f - track_cam_states[i].p_G))

            # Calculate predicted measurement at pose i of feature track j
            z_hat_i_j = np.array([[p_C_f[0, 0] / p_C_f[2, 0]],
                                  [p_C_f[1, 0] / p_C_f[2, 0]]])

            # Convert measurment to normalized pixel coordinates
            z = track.track[i].pt
            z = np.array([[(z[0] - cx) / fx],
                          [(z[1] - cy) / fy]])

            # Residual
            residual = z - z_hat_i_j
            r_j.append(residual)

            # Debug
            if debug:
                print()
                print("p_C_f:\n", p_C_f)
                print("z:\n", z)
                print("z_hat:\n", z_hat_i_j)

        return np.array(r_j).reshape((2 * len(track_cam_states), 1))

    def augment_state(self):
        """ Augment state

        Augment state and covariance matrix with a copy of the current camera
        pose estimate when a new image is recorded

        """
        # Using current IMU pose estimate to calculate camera pose
        cam_q_CG = quatmul(self.ext_q_CI, self.imu_state.q_IG)
        cam_p_G = self.imu_state.p_G + dot(C(self.imu_state.q_IG).T, self.ext_p_IC) # noqa

        # Camera pose Jacobian
        N = len(self.cam_states)
        J = self.J(self.ext_q_CI, self.ext_p_IC, self.imu_state.q_IG, N)

        # Build covariance matrix (without new camera state)
        P = self.P()

        # Augment MSCKF covariance matrix (with new camera state)
        X = np.block([[I(15 + 6 * N)], [J]])
        P = dot(X, dot(P, X.T))
        self.P_imu = P[0:15, 0:15]
        self.P_cam = P[15:, 15:]
        self.P_imu_cam = P[0:15, 15:]

        # Add new camera state to sliding window
        self.cam_states.append(CameraState(cam_q_CG, cam_p_G))

    def measurement_update(self):
        # # Build covariance matrix
        # P = self.P()
        #
        # # Calculate Kalman gain
        # K = dot(self.P(), T_H.T) * inv(dot(T_H, dot(self.P(), T_H.T)) + R_n)

        # # State correction
        # dX = K * r_n
        # msckfState = updateState(msckfState, dX)

        # # Covariance correction
        # tempMat = (eye(12 + 6 * size(msckfState.camStates, 2)) - K * T_H)
        # # tempMat = (eye(12 + 6*size(msckfState.camStates,2)) - K*H_o);
        #
        # P_corrected = tempMat * P * tempMat.T + K * R_n * K.T
        pass
