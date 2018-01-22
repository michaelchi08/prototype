from math import cos
from math import sin

import numpy as np
from numpy import dot
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
    R = None
    if orientation.shape == (3, 3):
        R = orientation
    else:
        R = euler2rot([deg2rad(i) for i in orientation], 123)
    p0 = dot(R, p0)
    p1 = dot(R, p1)

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
    points = np.array([dot(R, pt) for pt in points])

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

    def plot(self, ax):
        """ Plot gimbal

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Plot axes

        """
        self.attitude = np.array([deg2rad(25.0), deg2rad(30.0), 0.0])

        # Create base frame
        rpy = [-90.0, 0.0, -90.0]
        R_BG = euler2rot([deg2rad(i) for i in rpy], 321)
        t_G_B = np.array([0.0, 0.0, 0.0])
        T_GB = np.array([[R_BG[0, 0], R_BG[0, 1], R_BG[0, 2], t_G_B[0]],
                         [R_BG[1, 0], R_BG[1, 1], R_BG[1, 2], t_G_B[1]],
                         [R_BG[2, 0], R_BG[2, 1], R_BG[2, 2], t_G_B[2]],
                         [0.0, 0.0, 0.0, 1.0]])

        # Create DH Transforms
        T_B1 = dh_transform_matrix(self.attitude[0],
                                   deg2rad(0.0),
                                   self.roll_bar_width,
                                   0.0)
        T_12 = dh_transform_matrix(deg2rad(180.0),
                                   deg2rad(0.0),
                                   0.0,
                                   self.roll_bar_length)
        T_23 = dh_transform_matrix(deg2rad(0.0),
                                   self.attitude[1],
                                   self.pitch_bar_length,
                                   0.0)

        R_3C = euler2rot([deg2rad(i) for i in [0.0, 0.0, 180.0]], 321)
        t_B_C = np.array([0.0, 0.0, 0.1])
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

        # Plot first link
        ax.plot([T_GB[0, 3], links[0][0, 3]],
                [T_GB[1, 3], links[0][1, 3]],
                [T_GB[2, 3], links[0][2, 3]], '--', color="black")

        ax.plot([links[0][0, 3], links[1][0, 3]],
                [links[0][1, 3], links[1][1, 3]],
                [links[0][2, 3], links[1][2, 3]], '--', color="black")

        ax.plot([links[1][0, 3], links[2][0, 3]],
                [links[1][1, 3], links[2][1, 3]],
                [links[1][2, 3], links[2][2, 3]], '--', color="black")

        ax.plot([links[2][0, 3], links[3][0, 3]],
                [links[2][1, 3], links[3][1, 3]],
                [links[2][2, 3], links[3][2, 3]], '--', color="black")

        self.plot_coord_frame(ax, T_GB, length=0.05)
        self.plot_coord_frame(ax, T_G1, length=0.05)
        self.plot_coord_frame(ax, T_G2, length=0.05)
        self.plot_coord_frame(ax, T_G3, length=0.05)
        self.plot_coord_frame(ax, T_GC, length=0.05)

        # Draw roll motor
        motor_origin = T_GB[0:3, 3]
        motor_attitude = T_GB[0:3, 0:3]
        plot_3d_cylinder(ax, 0.04, 0.08, motor_origin, motor_attitude, "red")

        # motor_origin = T_G2[0:3, 3]
        # motor_attitude = np.array([0.0, 0.0, 0.0])
        # plot_3d_cylinder(ax, 0.04, 0.08, motor_origin, motor_attitude, "red")

    #     # self.draw_pitch_gimbal(ax)
    #     # self.draw_camera(ax)
