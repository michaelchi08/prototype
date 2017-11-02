import sympy
import numpy as np
from numpy.linalg import inv

from prototype.vision.geometry import triangulate_point


def quatmul(p, q):
    """ JPL Quaternion multiplication

    Source:

        Page 3.

        Trawny, Nikolas, and Stergios I. Roumeliotis. "Indirect Kalman filter
        for 3D attitude estimation." University of Minnesota, Dept. of Comp.
        Sci. & Eng., Tech. Rep 2 (2005): 2005.

    Args:

        p (np.array - 4x1): 1st Quaternion (x, y, z, w)
        q (np.array - 4x1): 2nd Quaternion (x, y, z, w)

    Return:

        Product of quaternion multiplication

    """
    p1, p2, p3, p4 = p.ravel()
    q1, q2, q3, q4 = q.ravel()

    return np.array([[q4 * p1 + q3 * p2 - q2 * p3 + q1 * p4],
                     [-q3 * p1 + q4 * p2 + q1 * p3 + q2 * p4],
                     [q2 * p1 - q1 * p2 + q4 * p3 + q3 * p4],
                     [-q1 * p1 - q2 * p2 - q3 * p3 + q4 * p4]])


def quat2rot(q):
    """ JPL Quaternion to rotation matrix

    Source:

        Page 9.

        Trawny, Nikolas, and Stergios I. Roumeliotis. "Indirect Kalman filter
        for 3D attitude estimation." University of Minnesota, Dept. of Comp.
        Sci. & Eng., Tech. Rep 2 (2005): 2005.

    Args:

        q (np.array - 4x1): 1st Quaternion (x, y, z, w)

    Return:

        Product of quaternion multiplication

    """
    q1, q2, q3, q4 = q.ravel()

    R11 = 1.0 - 2.0 * q2**2.0 - 2.0 * q3**2.0
    R12 = 2.0 * (q1 * q2 + q3 * q4)
    R13 = 2.0 * (q1 * q3 - q2 * q4)

    R21 = 2.0 * (q1 * q2 - q3 * q4)
    R22 = 1.0 - 2.0 * q1**2.0 - 2.0 * q3**2.0
    R23 = 2.0 * (q2 * q3 + q1 * q4)

    R31 = 2.0 * (q1 * q3 + q2 * q4)
    R32 = 2.0 * (q2 * q3 - q1 * q4)
    R33 = 1.0 - 2.0 * q1**2.0 - 2.0 * q2**2.0

    return np.matrix([[R11, R12, R13],
                      [R21, R22, R23],
                      [R31, R32, R33]])


def skew(v):
    """ Skew symmetric matrix

    Args:

        v (np.array): vector of size 3

    Returns:

        Skew symetric matrix (np.matrix)

    """
    return np.matrix([[0.0, -v[2], v[1]],
                      [v[2], 0.0, -v[0]],
                      [-v[1], v[0], 0.0]])


def C(q):
    """ Rotation matrix parameterized by a JPL quaternion (x, y, z, w)

    Args:

        q (np.array): Quaternion (x, y, z, w)

    Returns:

        Rotation matrix (np.matrix)

    """
    return np.matrix(quat2rot(q))


def wgn(mu, sigma):
    """ Gaussian White Noise

    Args:

        mu (float): Mean
        sigma (float): Variance


    Returns:

        Gaussian white noise as a float scalar value

    """
    return np.random.normal(mu, sigma, 1)[0]


def Omega(w):
    """ Omega function

    Args:

        w (np.array): Angular velocity

    Returns:

        Differential form of an angular velocity (np.array)

    """
    w = w.reshape((3, 1))
    return np.block([[-skew(w), w], [-w.T, 0.0]])


def zero(m, n):
    """ Zero matrix of size mxn

    Args:

        m (float): Number of rows
        n (float): Number of cols

    Returns:

        mxn zero matrix (np.matrix)

    """
    return np.matrix(np.zeros((m, n)))


def I(n):
    """ Return identity matrix of size nxn

    Args:

        n (float): Size of identity square matrix

    Returns:

        Identity matrix of size nxn (np.matrix)

    """
    return np.matrix(np.eye(n))


class CameraState:
    """ Camera state """

    def __init__(self, p_G_C, q_C_G):
        """ Constructor

        Args:

            p_G_C (np.array): Position of camera in Global frame
            q_C_G (np.array): Orientation of camera in Global frame

        """
        self.p_G_C = np.array(p_G_C).reshape((3, 1))
        self.q_C_G = np.array(q_C_G).reshape((4, 1))


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

        # IMU error state vector
        # X_imu = [q_I_G   # Orientation
        #          b_g     # Gyroscope Bias
        #          G_v_I   # Velocity
        #          b_a     # Accelerometer Bias
        #          G_p_I]  # Position
        self.X_imu = zero(5, 1)

        # IMU system noise vector
        self.n_imu = np.block([n_g.ravel(),
                               n_wg.ravel(),
                               n_a.ravel(),
                               n_wa.ravel()]).reshape((12, 1))

        # IMU covariance matrix
        self.Q_imu = I(12) * self.n_imu

        # Camera extrinsics
        self.cam_p_I_C = np.array([0.0, 0.0, 0.0])
        self.cam_q_C_I = np.array([1.0, 0.0, 0.0, 0.0])

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
        F = zero(15, 15)
        # -- First row --
        F[0:3, 0:3] = -skew(w_hat)
        F[0:3, 3:6] = np.ones((3, 3))
        # -- Third Row --
        F[6:9, 0:3] = -C(q_hat).T * skew(a_hat)
        F[6:9, 6:9] = -2.0 * skew(w_G)
        F[6:9, 9:12] = -C(q_hat).T
        F[6:9, 12:15] = -skew(w_G)**2
        # -- Fifth Row --
        F[12:15, 6:9] = np.ones((3, 3))

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
        G = zero(15, 12)
        # -- First row --
        G[0:3, 0:3] = np.ones((3, 3))
        # -- Second row --
        G[3:6, 3:6] = np.ones((3, 3))
        # -- Third row --
        G[6:9, 6:9] = -C(q_hat).T
        # -- Fourth row --
        G[9:12, 9:12] = np.ones((3, 3))

        return G

    def J(self, cam_q_C_I, cam_p_I_C, q_hat_I_G, N):
        """ Jacobian J matrix

        Args:

            cam_q_C_I (np.array): Rotation from IMU to camera frame
                                 in quaternion (x, y, z, w)
            cam_p_I_C (np.array): Position of camera frame from IMU
            q_hat_I_G (np.array): Rotation from global to IMU frame

        """
        q_hat_I_G, b_hat_g, v_hat_G_I, b_hat_a, p_hat_G_I = self.X_imu
        C_C_I = C(cam_q_C_I)
        C_I_G = C(q_hat_I_G)

        J = zero(6, 15 + 6 * N)

        # -- First row --
        J[0:3, 0:3] = C_C_I

        # -- Second row --
        J[3:6, 0:3] = skew(C_I_G.T * cam_p_I_C)
        J[3:6, 9:12] = I(3)

    def prediction_update(self, a_m, w_m, dt):
        """ IMU state update """
        w_G = np.array([0.0, 0.0, 1.0]).reshape((3, 1))
        G_g = np.array([0.0, 0.0, 1.0]).reshape((3, 1))

        # IMU error state estimates
        q_hat_I_G, b_hat_g, v_hat_G_I, b_hat_a, p_hat_G_I = self.X_imu

        # Calculate new accel and gyro estimates
        a_hat = a_m - b_hat_a * dt
        w_hat = w_m - b_hat_g - C(q_hat_I_G) * w_G * dt

        # Update IMU states (reverse order)
        p_hat_G_I = v_hat_G_I
        b_hat_a = b_hat_a + zero(3, 1)
        v_hat_G_I = v_hat_G_I + C(q_hat_I_G) * a_hat - 2 * skew(w_G) * v_hat_G_I - skew(w_G)**2 * G_p_I + G_g  # noqa
        b_hat_g = b_hat_g + zero(3, 1)
        q_hat_I_G = q_hat_I_G + 0.5 * Omega(w_hat) * q_hat_I_G
        self.X_imu = np.array(q_hat_I_G, b_hat_g, v_hat_G_I, b_hat_a, p_hat_G_I)
        self.X_imu = self.X_imu.reshape((15, 1))

        # Build the jacobians F and G
        F = self.F(w_hat, q_hat_I_G, a_hat, w_G)
        G = self.G(q_hat_I_G)

        # State transition matrix
        Phi = I(15) + F * dt

        # Update covariance matrices
        self.P_imu = Phi * self.P_imu + self.P_imu * Phi.T + G * self.Q_imu * G.T  # noqa
        self.P_cam = self.P_cam
        self.P_imu_cam = Phi * self.P_imu_cam

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
            r (np.array - size 2Nx1): Residual vector over all camera states

        """
        # Calculate initial estimate of 3D position
        # -- Calculate rotation and translation of camera 0 and 1
        C_C0_G = np.matrix(quat2rot(track_cam_states[0].q_C_G))
        C_C1_G = np.matrix(quat2rot(track_cam_states[1].q_C_G))
        p_G_C0 = track_cam_states[0].p_G_C.reshape((3, 1))
        p_G_C1 = track_cam_states[1].p_G_C.reshape((3, 1))
        # -- Set camera 0 as origin, work out rotation and translation of
        # -- camera 1 relative to to camera 0
        C_C0_C1 = C_C0_G * C_C1_G.T
        t_C0_C1C0 = C_C0_G * (p_G_C1 - p_G_C0)
        # -- Triangulate
        x1 = np.block([track.track[0].pt, 1.0])
        x2 = np.block([track.track[1].pt, 1.0])
        P1 = cam_model.P(np.eye(3), np.ones(3).reshape((3, 1)))
        P2 = cam_model.P(C_C0_C1, t_C0_C1C0.reshape((3, 1)))
        X = triangulate_point(x1, x2, P1, P2)

        # Create inverse depth params (these are to be optimized)
        alpha = X[0] / X[2]
        beta = X[1] / X[2]
        rho = 1.0 / X[2]

        # Gauss Newton optimization
        r_Jprev = float("inf")  # residual jacobian

        for k in range(10):
            N = len(track_cam_states)
            r = zero(2 * N, 1)
            J = zero(2 * N, 3)

            # Calculate residuals
            for i in range(N):
                # Get camera current rotation and translation
                C_Ci_G = np.matrix(quat2rot(track_cam_states[i].q_C_G))
                p_G_Ci = track_cam_states[i].p_G_C.reshape((3, 1))

                # Set camera 0 as origin, work out rotation and translation
                # of camera i relative to to camera 0
                C_Ci_C0 = C_Ci_G * C_C0_G.T
                t_Ci_CiC0 = C_Ci_G * (p_G_C0 - p_G_Ci)

                # Project estimated feature location to image plane
                h = (C_Ci_C0 * np.array([[alpha], [beta], [1]])) + (rho * t_Ci_CiC0)  # noqa

                # Calculate reprojection error
                # -- Camera intrinsics
                cx, cy = cam_model.K[0, 2], cam_model.K[1, 2]
                fx, fy = cam_model.K[0, 0], cam_model.K[1, 1]
                # -- Convert measurment to normalized pixel coordinates
                z = track.track[i].pt
                z = np.array([(z[0] - cx) / fx, (z[1] - cy) / fy])
                # -- Convert feature location to normalized pixel coordinates
                x = np.array([h[0, 0] / h[2, 0], h[1, 0] / h[2, 0]])
                # -- Reprojcetion error
                r[2 * i:(2 * (i + 1))] = np.array(z - x).reshape((2, 1))

                # Form the Jacobian
                drdalpha = np.array([
                    -C_Ci_G[0, 0] / h[2] + (h[0] / h[2]**2) * C_Ci_G[2, 0],
                    -C_Ci_G[1, 0] / h[2] + (h[1] / h[2]**2) * C_Ci_G[2, 0]
                ])
                drdbeta = np.array([
                    -C_Ci_G[0, 1] / h[2] + (h[0] / h[2]**2) * C_Ci_G[2, 1],
                    -C_Ci_G[1, 1] / h[2] + (h[1] / h[2]**2) * C_Ci_G[2, 1]
                ])
                drdrho = np.array([
                    -t_Ci_CiC0[0] / h[2] + (h[0] / h[2]**2) * t_Ci_CiC0[2],
                    -t_Ci_CiC0[1] / h[2] + (h[1] / h[2]**2) * t_Ci_CiC0[2]
                ])
                J[2 * i:(2 * (i + 1)), 0] = drdalpha.reshape((2, 1))
                J[2 * i:(2 * (i + 1)), 1] = drdbeta.reshape((2, 1))
                J[2 * i:(2 * (i + 1)), 2] = drdrho.reshape((2, 1))

            # Update esitmated params using Gauss Newton
            delta = np.linalg.inv(J.T * J) * J.T * r
            theta_km1 = np.array([alpha, beta, rho]).reshape((3, 1))
            theta_k = theta_km1 - delta
            alpha = theta_k[0, 0]
            beta = theta_k[1, 0]
            rho = theta_k[2, 0]

            # Check how fast the residuals are converging to 0
            r_Jnew = float(0.5 * r.T * r)
            if r_Jnew < 0.0001:
                break
            r_J = abs((r_Jnew - r_Jprev) / r_Jnew)
            r_Jprev = r_Jnew

            # Break loop if not making any progress
            if r_J < 0.0001:
                break

        # Debug
        if debug:
            print(k)
            print(alpha)
            print(beta)
            print(rho)

        # Convert estimated inverse depth params back to feature position in
        # global frame.  See (Eq.38, Mourikis2007 (A Multi-State Constraint
        # Kalman Filter for # Vision-aided Inertial Navigation)
        z = 1 / rho
        X = np.array([[alpha], [beta], [1.0]])
        C_C0_G = np.matrix(quat2rot(track_cam_states[0].q_C_G))
        p_G_C0 = track_cam_states[0].p_G_C
        p_G_f = z * C_C0_G.T * X + p_G_C0

        return (p_G_f, k, np.array(r))

    def calculate_track_residual(self, track, track_cam_states, p_G_f):
        """ Calculate the residual of a single feature track

        It is important to note the residual calculated in this function is
        different compared to the residual calculated when using
        `MSCKF.estimate_feature()`. During feature estimation, it is optimizing
        over the inverse depth params but the feature point estimated during the
        optimization is in the camera frame, **hence the residual was in the
        camera frame**.

        **Here we are calculating the residual in the global frame**

        Args:

            p_G_f (np.array - 3x1): Feature position in global frame
            track (FeatureTrack): A single feature track
            track_cam_states (list of CameraState): Camera states where feature
                                                    track was observed

        Returns:

            Residual vector (np.array)

        """
        r_j = []

        # Calculate residual vector
        for i in range(len(track_cam_states)):
            # Transform feature position from global to camera frame at pose i
            C_C_G = quat2rot(track_cam_states[i].q_C_G)
            p_C_f = C_C_G * (p_G_f - track_cam_states[i].p_G_C)

            # Calculate predicted measurement at pose i of feature track j
            zhat_i_j = np.array([[p_C_f[0] / p_C_f[2]], [p_C_f[1] / p_C_f[2]]])

            # Append to residual vector
            residual = track.track[i] - zhat_i_j
            r_j.append(residual)

        return np.array(r_j)

    def state_augmentation(self):
        # IMU error state estimates
        q_hat_I_G, b_hat_g, v_hat_G_I, b_hat_a, p_hat_G_I = self.X_imu

        # Using current IMU pose estimate to calculate camera pose
        # -- Camera rotation
        q_CG = quatmul(self.cam_q_C_I, q_hat_I_G)
        # -- Camera translation
        C_I_G = quat2rot(q_hat_I_G)
        p_G_C = p_hat_G_I + C_I_G.T * self.cam_p_I_C

        # Camera pose Jacobian
        N = 1
        J = self.J(self.cam_q_C_I, self.cam_p_I_C, q_hat_I_G, N)

        # Build covariance matrix (without new camera state)
        P = np.block([[self.P_imu, self.P_imu_cam],
                      [self.P_imu_cam.T, self.P_cam]])

        # Augment MSCKF covariance matrix (with new camera state)
        X = np.block([[I(15 + 6 * N)], [J]])
        P = X * P * X.T

        self.X_cam[N] = p_G_C
        self.X_cam[N] = q_CG

    def measurement_update(self):
        # # Build MSCKF covariance matrix
        # # self.P = np.block([msckfState.imuCovar, msckfState.imuCamCovar,
        #     #               msckfState.imuCamCovar', msckfState.camCovar])
        #
        # # Calculate Kalman gain
        # K = (self.P * T_H.T) * inv(T_H * self.P * T_H.T + R_n)
        #
        # # State correction
        # dX = K * r_n
        # msckfState = updateState(msckfState, dX)
        #
        # # Covariance correction
        # tempMat = (eye(12 + 6 * size(msckfState.camStates, 2)) - K * T_H)
        # # tempMat = (eye(12 + 6*size(msckfState.camStates,2)) - K*H_o);
        #
        # P_corrected = tempMat * P * tempMat.T + K * R_n * K.T
        pass
