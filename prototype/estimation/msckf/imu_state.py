from math import sqrt

import numpy as np
from numpy import ones
from numpy import zeros
from numpy import dot
from numpy import eye as I


from prototype.utils.linalg import enforce_psd
from prototype.utils.linalg import skew
from prototype.utils.linalg import skewsq
from prototype.utils.quaternion.jpl import quatlcomp
from prototype.utils.quaternion.jpl import quatnormalize
from prototype.utils.quaternion.jpl import C
from prototype.utils.quaternion.jpl import Omega


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
    g_G : np.array - 3x1
        Gravitational acceleration

    """

    def __init__(self, q_IG, b_g, v_G, b_a, p_G, n_imu):
        self.size = 15  # Number of elements in state vector

        # State vector
        self.q_IG = np.array(q_IG).reshape((4, 1))
        self.b_g = np.array(b_g).reshape((3, 1))
        self.v_G = np.array(v_G).reshape((3, 1))
        self.b_a = np.array(b_a).reshape((3, 1))
        self.p_G = np.array(p_G).reshape((3, 1))

        # Constants
        self.w_G = np.array([[0.0], [0.0], [0.0]])
        self.g_G = np.array([[0.0], [0.0], [-9.81]])

        # Covariance matrix
        self.P = I(self.size) * 1e-6
        self.Q = I(12) * n_imu

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
        F[6:9, 6:9] = dot(-2.0, skew(w_G))
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
        # a_hat = (a_m - self.b_a) * dt
        # w_hat = (w_m - self.b_g - dot(C(self.q_IG), self.w_G)) * dt
        a_hat = a_m
        w_hat = w_m

        # Propagate IMU states
        # -- Orientation
        self.q_IG = self.q_IG + dot(0.5 * Omega(w_hat), self.q_IG) * dt
        self.q_IG = quatnormalize(self.q_IG)
        # -- Gyro bias
        self.b_g = self.b_g + zeros((3, 1))
        # -- Velocity
        self.v_G = self.v_G + (dot(C(self.q_IG).T, a_hat) - 2 * dot(skew(self.w_G), self.v_G) - dot(skewsq(self.w_G), self.p_G)) * dt
        # self.v_G = self.v_G + (dot(C(self.q_IG).T, a_hat) - 2 * dot(skew(self.w_G), self.v_G) - dot(skewsq(self.w_G), self.p_G) + self.g_G) * dt  # noqa
        # -- Accel bias
        self.b_a = self.b_a + zeros((3, 1))
        # -- Position
        self.p_G = self.p_G + self.v_G * dt

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

    def __str__(self):
        s = "IMU state:\n"
        s += "q:\t{}\n".format(str(np.round(self.q_IG, 2).ravel()))
        s += "b_g:\t{}\n".format(str(np.round(self.b_g, 2).ravel()))
        s += "p:\t{}\n".format(str(np.round(self.p_G, 2).ravel()))
        s += "b_a:\t{}\n".format(str(np.round(self.b_a, 2).ravel()))
        s += "p_G:\t{}\n".format(str(np.round(self.p_G, 2).ravel()))
        return s
