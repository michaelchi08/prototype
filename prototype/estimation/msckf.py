import numpy as np

from prototype.utils.utils import quat2rot


def skew(v):
    """ Return skew symmetric matrix """
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def C(q) -> np.matrix:
    """ Return rotation matrix parameterized by a quaternion (w, x, y, z) """
    return np.matrix(quat2rot(q))


def wgn(mu, sigma):
    """ Return Gaussian White Noise """
    return np.random.normal(mu, sigma, 1)[0]


def omega(w):
    """ Return Omega """
    return np.block([[-skew(w), w] [-w, 0.0]])


class MSCKF:
    """ Multi-State Constraint Kalman Filter """

    def __init__(self, **kwargs):
        self.mu = kwargs.get("mu")

        self.X_imu = np.array([0.0, 0.0])

        self.S = kwargs.get("S")
        self.R = kwargs.get("R")
        self.Q = kwargs.get("Q")

        self.mu_p = kwargs.get("mu_p")
        self.S_p = kwargs.get("S_p")
        self.K = kwargs.get("K")

    def imu_state_update(self, dt):
        """ IMU state update """
        q_I_W, w_W = self.X_imu

        # update gyroscope
        b_g = 0.0
        n_g = wgn(0.0, 0.1)
        w = w_W + C(q_I_W) * w_W + b_g + n_g

        # update accelerometer
        b_a = 0.0
        n_a = wgn(0.0, 0.1)
        a = C(q_I_W) * (a_W - g_W + 2 * skew(w_W) * v_I + skew(w_W)**2 * p_I) + b_a + n_a  # noqa

        # q_I_W_dot = 0.5 * omega

    def prediction_update(self, u, dt):
        """ Prediction update """
        g = g_func(u, self.mu, dt)
        G = G_func(u, self.mu, dt)

        self.mu_p = g
        self.S_p = G * self.S * G.T + self.R

    def measurement_update(self, y, dt):
        """ Measurement update """
        h = h_func(self.mu_p)
        H = H_func(self.mu_p)

        K = self.S_p * H.T * (H * self.S_p * H.T).I + self.Q
        self.mu = self.mu_p + K * (y - h)
        self.S = (eye(len(self.mu)) - K * H) * self.S_p


# w = np.matrix([[0.0, 0.0, 0.0],
#               [0.0, 0.0, 0.0],
#               [0.0, 0.0, 0.0]])
#
# I = np.matrix([[0.0, 0.0, 0.0],
#               [0.0, 0.0, 0.0],
#               [0.0, 0.0, 0.0]])
