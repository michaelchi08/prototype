import numpy as np
from numpy import dot

from prototype.models.gimbal import GimbalModel


class PlotGimbal:
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
        self.gimbal = GimbalModel()

        self.link0 = None
        self.link1 = None
        self.link2 = None
        self.link3 = None

    def set_attitude(self, attitude):
        self.gimbal.attitude = attitude

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

    def plot(self, ax):
        """ Plot gimbal

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Plot axes

        """
        self.gimbal.set_attitude([20, 10, 0])
        links = self.gimbal.calc_transforms()

        # Plot links
        self.link0 = ax.plot([0, links[0][0, 3]],
                             [0, links[0][1, 3]],
                             [0, links[0][2, 3]],
                             '--', color="black")

        self.line1 = ax.plot([links[0][0, 3], links[1][0, 3]],
                             [links[0][1, 3], links[1][1, 3]],
                             [links[0][2, 3], links[1][2, 3]],
                             '--', color="black")

        self.line2 = ax.plot([links[1][0, 3], links[2][0, 3]],
                             [links[1][1, 3], links[2][1, 3]],
                             [links[1][2, 3], links[2][2, 3]],
                             '--', color="black")

        # Plot coordinate frames
        self.plot_coord_frame(ax, np.eye(4), length=0.05)
        self.plot_coord_frame(ax, links[0], length=0.05)
        self.plot_coord_frame(ax, links[1], length=0.05)
        self.plot_coord_frame(ax, links[2], length=0.05)
