from math import pi
from math import radians

import numpy as np
from numpy import dot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA

from prototype.utils.euler import euler2rot
from prototype.models.gimbal import GimbalModel
from prototype.viz.common import axis_equal_3dplot


class PlotGimbal:
    """ Gimbal plot

    Attributes:
    -----------
    origin : np.array
        Gimbal origin

    attitude : np.array
        Roll, pitch, yaw

    """
    def __init__(self, **kwargs):
        self.origin = np.array([0.0, 0.0, 0.0])
        self.gimbal = kwargs.get("gimbal", GimbalModel())

        self.link0 = None
        self.link1 = None
        self.link2 = None
        self.link3 = None

    def set_attitude(self, attitude):
        self.gimbal.set_attitude(attitude)

    def plot_coord_frame(self, ax, T, length=0.1):
        """ Plot coordinate frame

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Plot axes

        T : np.array (4x4)
            Transform matrix

        length : float (default: 0.1)
            Length of each coordinate frame axis

        """
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

    def plot(self, ax=None):
        """ Plot gimbal

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Plot axes

        """
        if ax is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')

        # Create transform from global origin to static camera
        t_g_sg = np.array([0.0, 0.0, 0.0])
        R_sg = euler2rot([-pi / 2.0, 0.0, -pi / 2.0], 321)
        T_gs = np.array([[R_sg[0, 0], R_sg[0, 1], R_sg[0, 2], t_g_sg[0]],
                         [R_sg[1, 0], R_sg[1, 1], R_sg[1, 2], t_g_sg[1]],
                         [R_sg[2, 0], R_sg[2, 1], R_sg[2, 2], t_g_sg[2]],
                         [0.0, 0.0, 0.0, 1.0]])

        # Calculate gimbal transforms
        T_sb, T_se, T_sd = self.gimbal.calc_transforms()
        T_gb = dot(T_gs, T_sb)
        T_ge = dot(T_gs, T_se)
        T_gd = dot(T_gs, T_sd)

        # Plot links
        self.link0 = ax.plot([0, T_gs[0, 3]],
                             [0, T_gs[1, 3]],
                             [0, T_gs[2, 3]],
                             '--', color="black")
        self.link1 = ax.plot([T_gs[0, 3], T_gb[0, 3]],
                             [T_gs[1, 3], T_gb[1, 3]],
                             [T_gs[2, 3], T_gb[2, 3]],
                             '--', color="black")
        self.link2 = ax.plot([T_gb[0, 3], T_ge[0, 3]],
                             [T_gb[1, 3], T_ge[1, 3]],
                             [T_gb[2, 3], T_ge[2, 3]],
                             '--', color="black")
        self.link3 = ax.plot([T_ge[0, 3], T_gd[0, 3]],
                             [T_ge[1, 3], T_gd[1, 3]],
                             [T_ge[2, 3], T_gd[2, 3]],
                             '--', color="black")

        # Plot coordinate frames
        self.plot_coord_frame(ax, T_gs, length=0.05)
        self.plot_coord_frame(ax, T_gb, length=0.05)
        self.plot_coord_frame(ax, T_ge, length=0.05)
        self.plot_coord_frame(ax, T_gd, length=0.05)

        # # Plot settings
        # if ax is None:
        #     axis_equal_3dplot(ax)
        #     ax.set_xlabel("x")
        #     ax.set_ylabel("y")
        #     ax.set_zlabel("z")

        return ax
