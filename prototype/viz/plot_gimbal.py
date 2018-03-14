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

        self.show_static_frame = kwargs.get("show_static_frame", True)
        self.show_base_frame = kwargs.get("show_base_frame", True)
        self.show_end_frame = kwargs.get("show_end_frame", True)
        self.show_dynamic_frame = kwargs.get("show_dynamic_frame", True)

    def set_attitude(self, attitude):
        self.gimbal.set_attitude(attitude)

    def plot_coord_frame(self, ax, T, frame, length=0.1):
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
        axis_x = dot(T, np.array([length, 0.0, 0.0, 1.0]))
        axis_y = dot(T, np.array([0.0, length, 0.0, 1.0]))
        axis_z = dot(T, np.array([0.0, 0.0, length, 1.0]))

        ax.text(T[0, 3], T[1, 3], T[2, 3] + 0.005, frame, color='Black')

        ax.plot([T[0, 3], axis_x[0]],
                [T[1, 3], axis_x[1]],
                [T[2, 3], axis_x[2]], color="red")
        ax.plot([T[0, 3], axis_y[0]],
                [T[1, 3], axis_y[1]],
                [T[2, 3], axis_y[2]], color="green")
        ax.plot([T[0, 3], axis_z[0]],
                [T[1, 3], axis_z[1]],
                [T[2, 3], axis_z[2]], color="blue")

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
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

        # Create transform from global origin to static camera
        t_g_sg = np.array([0.0, 0.0, 0.0])
        R_sg = euler2rot([-pi / 2.0, 0.0, -pi / 2.0], 123)
        T_sg = np.array([[R_sg[0, 0], R_sg[0, 1], R_sg[0, 2], t_g_sg[0]],
                         [R_sg[1, 0], R_sg[1, 1], R_sg[1, 2], t_g_sg[1]],
                         [R_sg[2, 0], R_sg[2, 1], R_sg[2, 2], t_g_sg[2]],
                         [0.0, 0.0, 0.0, 1.0]])

        # Calculate gimbal transforms
        T_bs, T_eb, T_de = self.gimbal.calc_transforms()

        # Plot static camera frame
        length = 0.1
        if self.show_static_frame:
            T_gs = np.linalg.inv(T_sg)
            self.plot_coord_frame(ax, T_gs, "S", length=length)

        # Plot base mechanism frame
        if self.show_base_frame:
            T_bg = dot(T_bs, T_sg)
            T_gb = np.linalg.inv(T_bg)
            self.plot_coord_frame(ax, T_gb, "B", length=length)

        # Plot end effector frame
        if self.show_end_frame:
            T_eg = dot(T_eb, dot(T_bs, T_sg))
            T_ge = np.linalg.inv(T_eg)
            self.plot_coord_frame(ax, T_ge, "E", length=length)

        # Plot dynamic camera frame
        if self.show_dynamic_frame:
            T_dg = dot(T_de, dot(T_eb, dot(T_bs, T_sg)))
            T_gd = np.linalg.inv(T_dg)
            self.plot_coord_frame(ax, T_gd, "D", length=length)

        # Plot links
        # self.link0 = ax.plot([0, T_gs[0, 3]],
        #                      [0, T_gs[1, 3]],
        #                      [0, T_gs[2, 3]],
        #                      '--', color="black")
        # self.link1 = ax.plot([T_gs[0, 3], T_gb[0, 3]],
        #                      [T_gs[1, 3], T_gb[1, 3]],
        #                      [T_gs[2, 3], T_gb[2, 3]],
        #                      '--', color="black")
        # self.link2 = ax.plot([T_gb[0, 3], T_ge[0, 3]],
        #                      [T_gb[1, 3], T_ge[1, 3]],
        #                      [T_gb[2, 3], T_ge[2, 3]],
        #                      '--', color="black")
        # self.link3 = ax.plot([T_ge[0, 3], T_gd[0, 3]],
        #                      [T_ge[1, 3], T_gd[1, 3]],
        #                      [T_ge[2, 3], T_gd[2, 3]],
        #                      '--', color="black")

        # Plot settings
        # if ax is None:
        #     axis_equal_3dplot(ax)
        axis_equal_3dplot(ax)

        return ax
