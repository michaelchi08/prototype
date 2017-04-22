#!/usr/bin/env python3
from numpy import eye
from numpy import dot
from numpy.linalg import inv


class KF(object):
    def __init__(self):
        self.mu = None
        self.S = None

        self.A = None
        self.B = None
        self.C = None

        self.R = None
        self.Q = None

        self.m = None
        self.S = None
        self.K = None

    # def prediction_update(self, u):
    #     self.mu_p = dot(self.A, self.mu) + dot(self.B, u)
    #     self.S_p = dot(self.A, dot(self.S, self.A.T)) + self.R
    #     self.K = self.S_p * self.C * inv(dot(self.C, dot(self.S_p, self.C.T)) + self.Q)
    #
    # def measurement_update(self, y):
    #     self.mu = self.mu_p + self.K * (y - self.C * self.mu_p)
    #     self.S = (eye(len(self.A)) - self.K * self.C) * self.S_p
    #
    # def estimate(self, u, y):
    #     self.prediction_update(u)
    #     self.measurement_update(y)
