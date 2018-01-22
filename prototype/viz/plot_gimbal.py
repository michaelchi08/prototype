from math import cos
from math import sin

import numpy as np
from numpy import dot
from scipy.linalg import norm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # NOQA

from prototype.utils.euler import euler2rot
from prototype.utils.utils import deg2rad


def dh_transform_matrix(theta, alpha, a, d):
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


class GimbalPlot:
    """ Gimbal plot

    Attributes:
    -----------
    origin : np.array
        Gimbal origin

    attitude : np.array
        Roll, pitch, yaw

    """
    def __init__(self):
        self.origin = np.array([0.0, 0.0, 0.0])
        self.attitude = np.array([deg2rad(0.0), deg2rad(0.0), 0.0])

        self.roll_bar_width = 0.5
        self.roll_bar_length = 0.5
        self.pitch_bar_length = 0.5

        self.link0 = None
        self.link1 = None
        self.link2 = None
        self.link3 = None

    def plot_coord_frame(self, ax, T, length=0.1):
        R = T[0:3, 0:3]
        t = T[0:3, 3]

        axis_x = dot(R, np.array([length, 0.0, 0.0])) + t
        axis_y = dot(R, np.array([0.0, length, 0.0])) + t
        axis_z = dot(R, np.array([0.0, 0.0, length])) + t

        ax.plot([t[0], axis_x[0]],
                [t[1], axis_x[1]],
                [t[2], axis_x[2]], color="red")
        ax.plot([t[0], axis_y[0]],
                [t[1], axis_y[1]],
                [t[2], axis_y[2]], color="green")
        ax.plot([t[0], axis_z[0]],
                [t[1], axis_z[1]],
                [t[2], axis_z[2]], color="blue")

    def set_attitude(self, attitude):
        self.attitude = attitude

    def calc_transforms(self):
        # Create base frame
        rpy = [-90.0, 0.0, -90.0]
        R_BG = euler2rot([deg2rad(i) for i in rpy], 321)
        t_G_B = np.array([0.0, 0.0, 0.0])
        T_GB = np.array([[R_BG[0, 0], R_BG[0, 1], R_BG[0, 2], t_G_B[0]],
                         [R_BG[1, 0], R_BG[1, 1], R_BG[1, 2], t_G_B[1]],
                         [R_BG[2, 0], R_BG[2, 1], R_BG[2, 2], t_G_B[2]],
                         [0.0, 0.0, 0.0, 1.0]])

        # Create DH Transforms
        roll, pitch, _ = self.attitude
        T_B1 = dh_transform_matrix(roll, 0.0, self.roll_bar_width, 0.0)
        T_12 = dh_transform_matrix(deg2rad(-90.0), deg2rad(90.0), 0.0, self.roll_bar_length)
        T_23 = dh_transform_matrix(deg2rad(90.0 + pitch), deg2rad(90.0), 0.0, self.pitch_bar_length)

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

    def plot(self, ax):
        """ Plot gimbal

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Plot axes

        """
        links, [T_GB, T_G1, T_G2, T_G3, T_GC] = self.calc_transforms()

        # Plot links
        self.link0 = ax.plot([T_GB[0, 3], links[0][0, 3]],
                             [T_GB[1, 3], links[0][1, 3]],
                             [T_GB[2, 3], links[0][2, 3]],
                             '--', color="black")

        self.line1 = ax.plot([links[0][0, 3], links[1][0, 3]],
                             [links[0][1, 3], links[1][1, 3]],
                             [links[0][2, 3], links[1][2, 3]],
                             '--', color="black")

        self.line2 = ax.plot([links[1][0, 3], links[2][0, 3]],
                             [links[1][1, 3], links[2][1, 3]],
                             [links[1][2, 3], links[2][2, 3]],
                             '--', color="black")

        self.line3 = ax.plot([links[2][0, 3], links[3][0, 3]],
                             [links[2][1, 3], links[3][1, 3]],
                             [links[2][2, 3], links[3][2, 3]],
                             '--', color="black")

        # Plot coordinate frames
        self.plot_coord_frame(ax, T_GB, length=0.05)
        self.plot_coord_frame(ax, T_G1, length=0.05)
        self.plot_coord_frame(ax, T_G2, length=0.05)
        self.plot_coord_frame(ax, T_G3, length=0.05)
        self.plot_coord_frame(ax, T_GC, length=0.05)
