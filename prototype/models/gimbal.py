from math import cos
from math import sin
from math import pi

import numpy as np
from numpy import dot

from prototype.utils.euler import euler2rot
from prototype.utils.utils import deg2rad


def dh_transform(theta, alpha, a, d):
    """ Denavit–Hartenberg transform matrix

    Parameters
    ----------
    theta : float
        Angle (radians)
    alpha : float
        Angle (radians)
    a : float
        Offset (m)
    d : float
        Offset (m)

    Returns
    -------
    DH Transform matrix

    """
    c = cos
    s = sin

    return np.array([
        [c(theta), -s(theta) * c(alpha), s(theta) * s(alpha), a * c(theta)],
        [s(theta), c(theta) * c(alpha), -c(theta) * s(alpha), a * s(theta)],
        [0.0, s(alpha), c(alpha), d],
        [0.0, 0.0, 0.0, 1.0],
    ])


class GimbalModel:
    def __init__(self):
        self.attitude = np.array([0.0, 0.0, 0.0])
        self.roll_bar_width = 0.5
        self.roll_bar_length = 0.5
        self.pitch_bar_length = 0.5

    def set_attitude(self, attitude):
        self.attitude = attitude

    def calc_transforms(self):
        # Create base frame
        R_BG = euler2rot([deg2rad(i) for i in [-90.0, 0.0, -90.0]], 321)
        t_G_B = np.array([0.0, 0.0, 0.0])
        T_GB = np.array([[R_BG[0, 0], R_BG[0, 1], R_BG[0, 2], t_G_B[0]],
                         [R_BG[1, 0], R_BG[1, 1], R_BG[1, 2], t_G_B[1]],
                         [R_BG[2, 0], R_BG[2, 1], R_BG[2, 2], t_G_B[2]],
                         [0.0, 0.0, 0.0, 1.0]])

        # Create DH Transforms
        roll, pitch, _ = self.attitude
        roll = deg2rad(roll)
        pitch = deg2rad(pitch)
        T_B1 = dh_transform(deg2rad(roll), 0.0, self.roll_bar_width, 0.0)
        T_12 = dh_transform(-pi / 2.0, pi / 2.0, 0.0, self.roll_bar_length)
        T_23 = dh_transform(pi / 2.0 + pitch, pi / 2.0, 0.0, self.pitch_bar_length) # NOQA

        R_3C = euler2rot([deg2rad(i) for i in [-90.0, 0.0, -90.0]], 321)
        t_B_C = np.array([0.1, 0.0, 0.0])
        T_3C = np.array([[R_3C[0, 0], R_3C[0, 1], R_3C[0, 2], t_B_C[0]],
                         [R_3C[1, 0], R_3C[1, 1], R_3C[1, 2], t_B_C[1]],
                         [R_3C[2, 0], R_3C[2, 1], R_3C[2, 2], t_B_C[2]],
                         [0.0, 0.0, 0.0, 1.0]])

        # Create transforms
        T_G1 = dot(T_GB, T_B1)
        T_G2 = dot(T_GB, dot(T_B1, T_12))
        T_G3 = dot(T_GB, dot(T_B1, dot(T_12, T_23)))
        T_GC = dot(T_GB, dot(T_B1, dot(T_12, dot(T_23, T_3C))))

        # Create links
        links = []
        links.append(T_G1)
        links.append(T_G2)
        links.append(T_G3)
        links.append(T_GC)

        return links, [T_GB, T_G1, T_G2, T_G3, T_GC]
