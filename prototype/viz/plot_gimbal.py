from math import cos
from math import sin

import numpy as np
from numpy.linalg import inv
from scipy.linalg import norm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection  # NOQA

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


def plot_3d_cylinder(ax, radius, height, origin, orientation, color):
    """ Plot 3D Cylinder

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Plot axes

    radius : float
        Radius

    height : float
        Height

    origin : np.array
        Origin

    orientation : np.array
        Orientation

    color : np.array
        Color

    """
    # Axis and radius
    p0 = np.array([0.0, 0.0, 0.0])  # Point at one end
    p1 = np.array([height, 0.0, 0.0])  # Point at other end

    # Rotate cylinder origin
    R = euler2rot([deg2rad(i) for i in orientation], 123)
    p0 = np.dot(R, p0)
    p1 = np.dot(R, p1)

    # Translate cylinder origin
    p0 += origin
    p1 += origin

    # Vector in direction of axis
    v = p1 - p0

    # Find magnitude of vector
    mag = norm(v)

    # Unit vector in direction of axis
    v = v / mag

    # Make some vector not in the same direction as v
    not_v = np.array([1, 0, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])

    # Make vector perpendicular to v
    n1 = np.cross(v, not_v)
    n1 /= norm(n1)

    # Make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)

    # Surface ranges over t from 0 to length of axis and 0 to 2*pi
    t = np.linspace(0, mag, 2)
    theta = np.linspace(0, 2 * np.pi, 100)
    rsample = np.linspace(0, radius, 2)

    # Use meshgrid to make 2d arrays
    t, theta2 = np.meshgrid(t, theta)
    rsample, theta = np.meshgrid(rsample, theta)

    # Generate coordinates for surface
    # "Tube"
    X, Y, Z = [p0[i] + v[i] * t + radius * np.sin(theta2) * n1[i] + radius * np.cos(theta2) * n2[i] for i in [0, 1, 2]]  # noqa
    # "Bottom"
    X2, Y2, Z2 = [p0[i] + rsample[i] * np.sin(theta) * n1[i] + rsample[i] * np.cos(theta) * n2[i] for i in [0, 1, 2]]  # noqa
    # "Top"
    X3, Y3, Z3 = [p0[i] + v[i] * mag + rsample[i] * np.sin(theta) * n1[i] + rsample[i] * np.cos(theta) * n2[i] for i in [0, 1, 2]]  # noqa

    ax.plot_surface(X, Y, Z, color=color)
    ax.plot_surface(X2, Y2, Z2, color=color)
    ax.plot_surface(X3, Y3, Z3, color=color)


def plot_3d_cube(ax, width, origin, orientation):
    """ Plot 3D cube

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Plot axes

    width : float
        Cube width

    origin : np.array
        Origin

    orientation : np.array
        Orientation

    """
    # Cube points
    points = np.array([[-1, -1, -1],
                       [1, -1, -1],
                       [1, 1, -1],
                       [-1, 1, -1],
                       [-1, -1, 1],
                       [1, -1, 1],
                       [1, 1, 1],
                       [-1, 1, 1]])
    points = points * (width / 2.0)

    # Rotate Cube
    R = euler2rot([deg2rad(i) for i in orientation], 123)
    points = np.array([np.dot(R, pt) for pt in points])

    # Translate Cube
    points += origin

    # Plot mesh grid
    r = [-1, 1]
    X, Y = np.meshgrid(r, r)

    # List of sides' polygons of figure
    verts = [[points[0], points[1], points[2], points[3]],
             [points[4], points[5], points[6], points[7]],
             [points[0], points[1], points[5], points[4]],
             [points[2], points[3], points[7], points[6]],
             [points[1], points[2], points[6], points[5]],
             [points[4], points[7], points[3], points[0]],
             [points[2], points[3], points[7], points[6]]]

    # Plot sides
    ax.add_collection3d(Poly3DCollection(verts,
                                         facecolors="black",
                                         linewidths=1,
                                         edgecolors="red",
                                         alpha=0.25))


class GimbalPlot:
    """ Gimbal plot

    Attributes:
    -----------
    origin : np.array
        Gimbal origin

    attitude : np.array
        Roll, pitch, yaw

    roll_motor_base : np.array
        Roll motor base position (x, y, z)
    roll_motor_attitude : np.array
        Roll motor attitude (roll, pitch, yaw)
    roll_bar_width : float
        Roll bar width
    roll_bar_length : float
        Roll bar_length

    pitch_motor_base : np.array
        Roll motor base position (x, y, z)
    pitch_motor_attitude : np.array
        Roll motor attitude (pitch, pitch, yaw)
    pitch_bar_length : float
        Roll bar_length

    camera_origin : np.array
        Camera position (x, y, z)
    camera_attitude : np.array
        Camera attitude (roll, pitch, yaw)

    """

    def __init__(self):
        self.origin = np.array([0.0, 0.0, 0.0])
        self.attitude = np.array([deg2rad(0.0), deg2rad(0.0), 0.0])

        self.roll_motor_base = np.array([0.0, 0.0, -0.3])
        self.roll_motor_attitude = np.array([0.0, 0.0, 0.0])
        self.roll_bar_width = 0.2
        self.roll_bar_length = 0.2

        self.pitch_motor_base = np.array([0.25, 0.25, -0.3])
        self.pitch_motor_attitude = np.array([0.0, 0.0, -90.0])
        self.pitch_bar_length = 0.2

        self.camera_origin = np.array([0.3, 0.0, -0.3])
        self.camera_attitude = np.array([0.0, 0.0, 0.0])

    def draw_roll_gimbal(self, ax, motor_base):
        """ Draw roll gimbal

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Plot axes

        """
        # Draw origin to roll motor
        ax.plot([self.origin[0], self.roll_motor_base[0]],
                [self.origin[1], self.roll_motor_base[1]],
                [self.origin[2], self.roll_motor_base[2]],
                linewidth=2.0,
                zorder=100,
                marker="o",
                color="blue")

        # Draw roll motor
        plot_3d_cylinder(ax,
                         0.04,
                         0.08,
                         self.roll_motor_base,
                         self.roll_motor_attitude,
                         "red")

        # # Draw roll bar
        # ax.plot([self.roll_motor_base[0], self.roll_motor_base[0]],
        #         [self.roll_motor_base[1], self.pitch_motor_base[1]],
        #         [self.roll_motor_base[2], self.pitch_motor_base[2]],
        #         linewidth=2.0,
        #         zorder=100,
        #         marker="o",
        #         color="blue")
        #
        # ax.plot([self.roll_motor_base[0], self.pitch_motor_base[0]],
        #         [self.pitch_motor_base[1], self.pitch_motor_base[1]],
        #         [self.pitch_motor_base[2], self.pitch_motor_base[2]],
        #         linewidth=2.0,
        #         zorder=100,
        #         marker="o",
        #         color="blue")

    def draw_pitch_gimbal(self, ax):
        """ Draw pitch gimbal

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Plot axes

        """
        plot_3d_cylinder(ax,
                         0.04,
                         0.08,
                         self.pitch_motor_base,
                         self.pitch_motor_attitude,
                         "green")

    def draw_camera(self, ax):
        """ Draw camera

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Plot axes

        """
        plot_3d_cube(ax, 0.1, self.camera_origin, self.camera_attitude)

    def plot(self, ax):
        """ Plot gimbal

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Plot axes

        """
        # # End effector to camera frame
        # rpy = [0.0, 0.0, 0.0]
        # R_c = euler2rot([deg2rad(i) for i in rpy], 123).reshape(3, 3)
        # t_c = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
        # camera = np.matrix([[R_c[0, 0], R_c[0, 1], R_c[0, 2], t_c[0, 0]],
        #                     [R_c[1, 0], R_c[1, 1], R_c[1, 2], t_c[1, 0]],
        #                     [R_c[2, 0], R_c[2, 1], R_c[2, 2], t_c[2, 0]],
        #                     [0.0, 0.0, 0.0, 1.0]])

        self.attitude = np.array([deg2rad(10.0), deg2rad(30.0), 0.0])

        # Base frame to first link
        rpy = [0.0, 90.0, 90.0]
        R_B0 = euler2rot([deg2rad(i) for i in rpy], 123).reshape(3, 3)
        t_0_B = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
        T_B0 = np.matrix([[R_B0[0, 0], R_B0[0, 1], R_B0[0, 2], t_0_B[0, 0]],
                          [R_B0[1, 0], R_B0[1, 1], R_B0[1, 2], t_0_B[1, 0]],
                          [R_B0[2, 0], R_B0[2, 1], R_B0[2, 2], t_0_B[2, 0]],
                          [0.0, 0.0, 0.0, 1.0]])

        # Create DH Transforms
        T_1B = dh_transform_matrix(self.attitude[0],
                                   deg2rad(0.0),
                                   self.roll_bar_width,
                                   0.0)
        T_21 = dh_transform_matrix(deg2rad(180.0),
                                  deg2rad(0.0),
                                  0.0,
                                  self.roll_bar_length)
        T_32 = dh_transform_matrix(deg2rad(0.0),
                                    self.attitude[1],
                                    self.pitch_bar_length,
                                    0.0)

        # Transform from camera to end-effector
        # T_10 = T_1B * T_B0      # Roll bar
        # T_21 = T_1 * T_2 * T_B0 # Pitch bar
        # T_ = T_1 * T_2 * T_3  # Pitch motor to camera

        # links = []
        # links.append(np.array(T_10[0:3, 3]).reshape(3, 1))
        # links.append(np.array(T_12[0:3, 3]).reshape(3, 1))
        # links.append(np.array(T_13[0:3, 3]).reshape(3, 1))

        # Plot first link
        # ax.plot([t_0_B[0, 0], links[0][0, 0]],
        #         [t_0_B[1, 0], links[0][1, 0]],
        #         [t_0_B[2, 0], links[0][2, 0]], color="blue")

        # ax.plot([links[0][0, 0], links[1][0, 0]],
        #         [links[0][1, 0], links[1][1, 0]],
        #         [links[0][2, 0], links[1][2, 0]], color="blue")
        #
        # ax.plot([links[1][0, 0], links[2][0, 0]],
        #         [links[1][1, 0], links[2][1, 0]],
        #         [links[1][2, 0], links[2][2, 0]], color="blue")

        # Plot end effector coordinate frame
        # R = T_13[0:3, 0:3]
        # t = T_13[0:3, 3]
        # axis_x = R * np.array([[0.1], [0.0], [0.0]]) + t
        # axis_y = R * np.array([[0.0], [0.1], [0.0]]) + t
        # axis_z = R * np.array([[0.0], [0.0], [0.1]]) + t
        #
        # ax.plot([t[0, 0], axis_x[0, 0]],
        #         [t[1, 0], axis_x[1, 0]],
        #         [t[2, 0], axis_x[2, 0]], color="red")
        #
        # ax.plot([t[0, 0], axis_y[0, 0]],
        #         [t[1, 0], axis_y[1, 0]],
        #         [t[2, 0], axis_y[2, 0]], color="green")
        #
        # ax.plot([t[0, 0], axis_z[0, 0]],
        #         [t[1, 0], axis_z[1, 0]],
        #         [t[2, 0], axis_z[2, 0]], color="blue")

        # self.draw_roll_gimbal(ax, links[0])
        # self.draw_pitch_gimbal(ax)
        # self.draw_camera(ax)
