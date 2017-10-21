import numpy as np

from prototype.utils.utils import quat2rot
from prototype.vision.geometry import triangulate


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
    """ Rotation matrix parameterized by a quaternion (w, x, y, z)

    Args:

        q (np.array): Quaternion (w, x, y, z)

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
    return np.block([[-skew(w), w], [-w, 0.0]])


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
    def __init__(self, p_G_C, q_GC):
        """ Constructor

        Args:

            p_G_C (np.array): Position of camera in Global frame
            q_G_C (np.array): Orientation of camera in Global frame

        """
        self.p_G_C = p_G_C
        self.q_GC = q_GC


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
        self.cam_q_CI = np.array([1.0, 0.0, 0.0, 0.0])

    def F(self, w_hat, q_hat, a_hat, w_G):
        """ Transition Jacobian F matrix

        Aka predicition or transition matrix in an EKF

        Args:

            w_hat (np.array): Estimated angular velocity
            q_hat (np.array): Estimated quaternion (w, x, y, z)
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

            q_hat (np.array): Estimated quaternion (w, x, y, z)

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

    def J(self, cam_q_CI, cam_p_I_C, q_hat_IG, N):
        """ Jacobian J matrix

        Args:

            cam_q_CI (np.array): Rotation from IMU to camera frame
                                 in quaternion (w, x, y, z)
            cam_p_I_C (np.array): Position of camera frame from IMU
            q_hat_IG (np.array): Rotation from global to IMU frame

        """
        q_hat_IG, b_hat_g, v_hat_G_I, b_hat_a, p_hat_G_I = self.X_imu
        C_CI = C(cam_q_CI)
        C_IG = C(q_hat_IG)

        J = zero(6, 15 + 6 * N)

        # -- First row --
        J[0:3, 0:3] = C_CI

        # -- Second row --
        J[3:6, 0:3] = skew(C_IG.T * cam_p_I_C)
        J[3:6, 9:12] = I(3)

    def prediction_update(self, a_m, w_m, dt):
        """ IMU state update """
        w_G = np.array([0.0, 0.0, 1.0]).reshape((3, 1))
        G_g = np.array([0.0, 0.0, 1.0]).reshape((3, 1))

        # IMU error state estimates
        q_hat_IG, b_hat_g, v_hat_G_I, b_hat_a, p_hat_G_I = self.X_imu

        # Calculate new accel and gyro estimates
        a_hat = a_m - b_hat_a * dt
        w_hat = w_m - b_hat_g - C(q_hat_IG) * w_G * dt

        # Update IMU states (reverse order)
        p_hat_G_I = v_hat_G_I
        b_hat_a = b_hat_a + zero(3, 1)
        v_hat_G_I = v_hat_G_I + C(q_hat_IG) * a_hat - 2 * skew(w_G) * v_hat_G_I - skew(w_G)**2 * G_p_I + G_g  # noqa
        b_hat_g = b_hat_g + zero(3, 1)
        q_hat_IG = q_hat_IG + 0.5 * Omega(w_hat) * q_hat_IG
        self.X_imu = np.array(q_hat_IG, b_hat_g, v_hat_G_I, b_hat_a, p_hat_G_I)
        self.X_imu = self.X_imu.reshape((15, 1))

        # Build the jacobians F and G
        F = self.F(w_hat, q_hat_IG, a_hat, w_G)
        G = self.G(q_hat_IG)

        # State transition matrix
        Phi = I(15) + F * dt

        # Update covariance matrices
        self.P_imu = Phi * self.P_imu + self.P_imu * Phi.T + G * self.Q_imu * G.T  # noqa
        self.P_cam = self.P_cam
        self.P_imu_cam = Phi * self.P_imu_cam

    def state_augmentation(self):
        # IMU error state estimates
        q_hat_IG, b_hat_g, v_hat_G_I, b_hat_a, p_hat_G_I = self.X_imu

        # Using current IMU pose estimate to calculate camera pose
        # -- Camera rotation
        q_CG = quatmul(self.cam_q_CI, q_hat_IG)
        # -- Camera translation
        C_IG = quat2rot(q_hat_IG)
        p_G_C = p_hat_G_I + C_IG.T * self.cam_p_I_C

        # Camera pose Jacobian
        N = 1
        J = self.J(self.cam_q_CI, self.cam_p_I_C, q_hat_IG, N)

        # Build covariance matrix (without new camera state)
        P = np.block([[self.P_imu, self.P_imu_cam],
                      [self.P_imu_cam.T, self.P_cam]])

        # Augment MSCKF covariance matrix (with new camera state)
        X = np.block([[I(15 + 6 * N)], [J]])
        P = X * P * X.T

        self.X_cam[N] = p_G_C
        self.X_cam[N] = q_CG

    def estimate_feature(self, cam_model, cam_states, track, noise_params):
        """ Estimate feature 3D location by optimizing over inverse depth
        paramterization using Gauss Newton Optimization

        Args:

            cam_states (list of CameraState): Camera states
            track (FeatureTrack): Feature track
            K (np.array): Camera intrinsics
            noise_params (np.array): Camera noise parameters

        Returns:

            Estimated feature 3D location

        """
        N = len(cam_states)

        # Calculate relative rotation and translation from Cam 0 to Cam 1
        C_C0_G = quat2rot(cam_states[0].q_CG)
        C_C1_G = quat2rot(cam_states[1].q_CG)
        C_C1_C0 = C_C1_G * C_C0_G.T
        t_G_C0 = cam_states[0].p_G_C
        t_C0_C1C0 = C_C1_G * (t_G_C0 - cam_states[1].p_G_C)

        # Calculate initial estimate of 3D position
        x1 = track.track[0]
        x2 = track.track[1]
        P1 = cam_model.P(np.eye(3), np.zeros((3, 1)))  # Set Cam 0 as origin
        P2 = cam_model.P(C_C1_C0, t_C0_C1C0)
        X = triangulate(x1, x2, P1, P2)

        # Inverse depth parameterization
        alpha = X[0] / X[2]
        beta = X[1] / X[2]
        rho = 1.0 / X[2]

        # Gauss Newton optimization
        N = len(cam_states)
        for i in range(N):
            # Calculate relative rotation and translation from Cam 0 to Cam 1
            C_Ci_G = quat2rot(cam_states[i].q_CG)
            C_Ci_C0 = C_Ci_G * C_C0_G.T
            t_C0_CiC0 = C_Ci_C0 * (t_G_C0 - cam_states[i].p_G_C)

            h = C_Ci_C0 * np.array([alpha, beta, 1.0]) + rho * t_C0_CiC0
