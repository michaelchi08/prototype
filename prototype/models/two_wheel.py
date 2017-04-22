#!/usr/bin/env python3
from math import cos
from math import sin

import numpy as np


def two_wheel_2d_model(x, u, dt):
    g1 = x[0] + u[0] * cos(x[2]) * dt
    g2 = x[1] + u[0] * sin(x[2]) * dt
    g3 = x[2] + u[1] * dt

    return np.array([g1, g2, g3])


def two_wheel_2d_linearized_model(x, u, dt):
    G1 = 1.0
    G2 = 0.0
    G3 = -u[0] * sin(x[2]) * dt

    G4 = 0.0
    G5 = 1.0
    G6 = u[0] * cos(x[2]) * dt

    G7 = 0.0
    G8 = 0.0
    G9 = 1.0

    G = [[G1, G2, G3],
         [G4, G5, G6],
         [G7, G8, G9]]

    return np.array(G)


def two_wheel_3d_model(x, u, dt):
    g1 = x[0] + u[0] * cos(x[3]) * dt
    g2 = x[1] + u[0] * sin(x[3]) * dt
    g3 = x[2] + u[1] * dt
    g4 = x[3] + u[2] * dt

    return np.array([g1, g2, g3, g4])
