from numpy import eye
from numpy import dot
from numpy.linalg import inv


class EKF(object):
    """ Extended Kalman Filter """

    def __init__(self):
        self.dt = None

        self.mu = None
        self.S = None
        self.R = None
        self.Q = None

        self.mu_p = None
        self.S_p = None
        self.K = None

        self.g_func = None
        self.G_func = None

        self.h_func = None
        self.H_func = None

    def prediction_update(self, g_func, G_func, u):
        """ Prediction update """
        g = g_func(u, self.mu, self.dt)
        G = G_func(u, self.mu, self.dt)

        self.mu_p = g
        self.S_p = dot(G, dot(self.S, G.T)) + self.R

    def measurement_update(self, h_func, H_func, y):
        """ Measurement update """
        h = h_func(self.mu_p)
        H = H_func(self.mu_p)

        K = dot(self.S_p, dot(H.T, inv(dot(H, dot(self.S_p, H.T)))) + self.Q)
        self.mu = self.mu_p + dot(K, (y - h))
        self.S = dot((eye(len(self.mu)) - dot(K, H)), self.S_p)
