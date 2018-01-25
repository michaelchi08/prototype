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
        self.width = 0.0
        self.length = 0.5

    def set_attitude(self, attitude):
        self.attitude = attitude

    def T_sb(self):
        # Create base frame
        R_sb = euler2rot([deg2rad(i) for i in [180.0, -90.0, 0.0]], 321)
        t_G_sb = np.array([0.0, 0.0, -1.0])
        T_sb = np.array([[R_sb[0, 0], R_sb[0, 1], R_sb[0, 2], t_G_sb[0]],
                         [R_sb[1, 0], R_sb[1, 1], R_sb[1, 2], t_G_sb[1]],
                         [R_sb[2, 0], R_sb[2, 1], R_sb[2, 2], t_G_sb[2]],
                         [0.0, 0.0, 0.0, 1.0]])

        return T_sb

    def T_be(self):
        # Create DH Transforms (theta, alpha, a, d)
        roll, pitch, _ = self.attitude
        roll = deg2rad(roll)
        pitch = deg2rad(pitch)
        T_b1 = dh_transform(roll, -pi / 2.0, 0.0, self.length)
        T_1e = dh_transform(-pitch, pi, 0, self.width)
        T_be = dot(T_b1, T_1e)
        return T_be

    def T_ed(self):
        R_ed = euler2rot([deg2rad(i) for i in [0.0, 90.0, 90.0]], 321)
        t = np.array([0.0, 0.1, 0.0])
        T_ed = np.array([[R_ed[0, 0], R_ed[0, 1], R_ed[0, 2], t[0]],
                         [R_ed[1, 0], R_ed[1, 1], R_ed[1, 2], t[1]],
                         [R_ed[2, 0], R_ed[2, 1], R_ed[2, 2], t[2]],
                         [0.0, 0.0, 0.0, 1.0]])

        return T_ed

    def calc_transforms(self):
        T_sb = self.T_sb()  # Transform from static camera to base frame
        T_be = self.T_be()  # Transform from base frame to end-effector
        T_ed = self.T_ed()  # Transform from end-effector to dynamic camera

        # Create transforms
        T_se = dot(T_sb, T_be)  # Transform static camera to end effector
        T_sd = dot(T_se, T_ed)  # Transform static camera to dynamic camera

        # Create links
        links = [T_sb, T_se, T_sd]

        return links
