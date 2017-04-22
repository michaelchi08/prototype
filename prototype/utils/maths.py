#!/usr/bin/env python3
from math import pi
from math import cos
from math import sin

import numpy as np


def deg2rad(d):
    return d * (pi / 180.0)


def rad2deg(r):
    return r * (180.0 / pi)


def rotx(theta):
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, cos(theta), sin(theta)],
                     [0.0, -sin(theta), cos(theta)]])


def roty(theta):
    return np.array([[cos(theta), 0.0, -sin(theta)],
                     [0.0, 1.0, 0.0],
                     [sin(theta), 0.0, cos(theta)]])


def rotz(theta):
    return np.array([[cos(theta), sin(theta), 0.0],
                     [-sin(theta), cos(theta), 0.0],
                     [0.0, 0.0, 1.0]])


def euler123(phi, theta, psi):
    R11 = cos(theta) * cos(psi)
    R12 = cos(theta) * sin(psi)
    R13 = -sin(theta)

    R21 = sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi)
    R22 = sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi)
    R23 = sin(phi) * cos(theta)

    R31 = cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)
    R32 = cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi)
    R33 = cos(phi) * cos(theta)

    return np.array([[R11, R12, R13],
                     [R21, R22, R23],
                     [R31, R32, R33]])


def euler321(phi, theta, psi):
    R11 = cos(theta) * cos(psi)
    R12 = sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi)
    R13 = cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)

    R21 = cos(theta) * sin(psi)
    R22 = sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi)
    R23 = cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi)

    R31 = -sin(theta)
    R32 = sin(phi) * cos(theta)
    R33 = cos(phi) * cos(theta)

    return np.array([[R11, R12, R13],
                     [R21, R22, R23],
                     [R31, R32, R33]])
