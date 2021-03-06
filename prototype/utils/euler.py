from math import cos
from math import sin

import numpy as np


def rotx(theta):
    """Rotation matrix around x-axis (counter-clockwise, right-handed)

    Parameters
    ----------
    theta : float
        Rotation around x-axis in radians

    Returns
    -------

        Rotation matrix (np.array)

    """
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, cos(theta), sin(theta)],
                     [0.0, -sin(theta), cos(theta)]])


def roty(theta):
    """Rotation matrix around y-axis (counter-clockwise, right-handed)

    Parameters
    ----------
    theta : float
        Rotation around y-axis in radians

    Returns
    -------

        Rotation matrix (np.array)

    """
    return np.array([[cos(theta), 0.0, -sin(theta)],
                     [0.0, 1.0, 0.0],
                     [sin(theta), 0.0, cos(theta)]])


def rotz(theta):
    """Rotation matrix around z-axis (counter-clockwise, right-handed)

    Parameters
    ----------
    theta : float
        Rotation around z-axis in radians

    Returns
    -------

        Rotation matrix (np.array)

    """
    return np.array([[cos(theta), sin(theta), 0.0],
                     [-sin(theta), cos(theta), 0.0],
                     [0.0, 0.0, 1.0]])


def euler2rot(euler, euler_seq):
    """Convert euler to rotation matrix R
    This function assumes we are performing a body fixed intrinsic rotation.

    Source:

        Kuipers, Jack B. Quaternions and Rotation Sequences: A Primer with
        Applications to Orbits, Aerospace, and Virtual Reality. Princeton, N.J:
        Princeton University Press, 1999. Print.

        Page 86.

    Parameters
    ----------
    euler : np.array
        Euler angle (roll, pitch, yaw)
    euler_seq : float
        Euler rotation sequence

    Returns
    -------

        Rotation matrix (np.array)

    """
    if euler_seq == 321:  # i.e. ZYX rotation sequence (world to body)
        phi, theta, psi = euler

        R11 = cos(psi) * cos(theta)
        R21 = sin(psi) * cos(theta)
        R31 = -sin(theta)

        R12 = cos(psi) * sin(theta) * sin(phi) - sin(psi) * cos(phi)
        R22 = sin(psi) * sin(theta) * sin(phi) + cos(psi) * cos(phi)
        R32 = cos(theta) * sin(phi)

        R13 = cos(psi) * sin(theta) * cos(phi) + sin(psi) * sin(phi)
        R23 = sin(psi) * sin(theta) * cos(phi) - cos(psi) * sin(phi)
        R33 = cos(theta) * cos(phi)

    elif euler_seq == 123:  # i.e. XYZ rotation sequence (body to world)
        phi, theta, psi = euler

        R11 = cos(psi) * cos(theta)
        R21 = cos(psi) * sin(theta) * sin(phi) - sin(psi) * cos(phi)
        R31 = cos(psi) * sin(theta) * cos(phi) + sin(psi) * sin(phi)

        R12 = sin(psi) * cos(theta)
        R22 = sin(psi) * sin(theta) * sin(phi) + cos(psi) * cos(phi)
        R32 = sin(psi) * sin(theta) * cos(phi) - cos(psi) * sin(phi)

        R13 = -sin(theta)
        R23 = cos(theta) * sin(phi)
        R33 = cos(theta) * cos(phi)

    else:
        err_msg = "Error! Unsupported euler sequence [%s]" % str(euler_seq)
        raise RuntimeError(err_msg)

    return np.array([[R11, R12, R13],
                     [R21, R22, R23],
                     [R31, R32, R33]])
