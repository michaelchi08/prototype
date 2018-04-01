from math import pi
from math import cos
from math import sin
from math import fmod

import numpy as np


def deg2rad(d):
    """ Convert degrees to radians

    Parameters
    ----------
    d (float): degrees

    Returns
    -------
    Radians

    """
    return d * (pi / 180.0)


def rad2deg(r):
    """ Convert radians to degrees

    Parameters
    ----------
    r (float): Radians

    Returns
    -------
    Degrees

    """
    return r * (180.0 / pi)


def wrap180(euler_angle):
    """ Wrap angle to 180

    Parameters
    ----------
    euler_angle (float): Euler angle

    Returns
    -------
    Wrapped angle

    """
    return fmod((euler_angle + 180.0), 360.0) - 180.0


def wrap360(euler_angle):
    """ Wrap angle to 360

    Parameters
    ----------
    euler_angle (float): Euler angle

    Returns
    -------
    Wrapped angle

    """
    if euler_angle > 0.0:
        return fmod(euler_angle, 360.0)
    else:
        euler_angle += 360.0
        return fmod(euler_angle, 360.0)


def rotx(theta):
    """ Rotation matrix around x-axis (counter-clockwise)

    Parameters
    ----------
    theta (float): Rotation around x in radians

    Returns
    -------
    3 x 3 Rotation matrix

    """
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, cos(theta), -sin(theta)],
                     [0.0, sin(theta), cos(theta)]])


def roty(theta):
    """ Rotation matrix around y-axis (counter-clockwise)

    Parameters
    ----------
    theta (float): Rotation around y in radians

    Returns
    -------
    3 x 3 Rotation matrix

    """
    return np.array([[cos(theta), 0.0, sin(theta)],
                     [0.0, 1.0, 0.0],
                     [-sin(theta), 0.0, cos(theta)]])


def rotz(theta):
    """ Rotation matrix around z-axis (counter-clockwise)

    Parameters
    ----------
    theta (float): Rotation around y in radians

    Returns
    -------
    3 x 3 Rotation matrix

    """
    return np.array([[cos(theta), -sin(theta), 0.0],
                     [sin(theta), cos(theta), 0.0],
                     [0.0, 0.0, 1.0]])


def rotnormalize(R):
    """Normalize rotation matrix R by enforcing det(R) = 1 by finding the
    nearest orthogonal matrix

    Parameters
    ----------
    R : np.array - 3x3
        Rotation matrix

    Returns
    -------
    R : np.array - 3x3
        Unitary rotation matrix

    """
    U, S, V = np.linalg.svd(R)
    R = np.dot(U, np.dot(np.eye(S.shape[0]), V))
    return R


def enu2nwu(enu):
    """ Convert vector in ENU to NWU coordinate system

    Parameters
    ----------
        enu (np.array or list of size 3)

    Returns
    -------
    nwu (np.array or list of size 3)

    """
    # ENU frame:  (x - right, y - forward, z - up)
    # NWU frame:  (x - forward, y - left, z - up)
    nwu = [0, 0, 0]
    nwu[0] = enu[1]
    nwu[1] = -enu[0]
    nwu[2] = enu[2]
    return nwu


def edn2nwu(edn):
    """ Convert vector in EDN to NWU coordinate system

    Parameters
    ----------
    edn (np.array or list of size 3)

    Returns
    -------
    nwu (np.array or list of size 3)

    """
    # camera frame:  (x - right, y - down, z - forward)
    # NWU frame:  (x - forward, y - left, z - up)
    nwu = [0, 0, 0]
    nwu[0] = edn[2]
    nwu[1] = -edn[0]
    nwu[2] = -edn[1]
    return nwu


def edn2enu(edn):
    """ Convert vector in EDN to ENU coordinate system

    Parameters
    ----------
    edn (np.array or list of size 3)

    Returns
    -------
    enu (np.array or list of size 3)

    """
    # camera frame:  (x - right, y - down, z - forward)
    # ENU frame:  (x - right, y - forward, z - up)
    enu = [0, 0, 0]
    enu[0] = edn[0]
    enu[1] = edn[2]
    enu[2] = -edn[1]
    return enu


def ned2enu(ned):
    """ Convert vector in NED to ENU coordinate system

    Parameters
    ----------
    ned (np.array or list of size 3)

    Returns
    -------
    enu (np.array or list of size 3)

    """
    # NED frame:  (x - forward, y - right, z - down)
    # ENU frame:  (x - right, y - forward, z - up)
    enu = [0, 0, 0]
    enu[0] = ned[1]
    enu[1] = ned[0]
    enu[2] = -ned[2]
    return enu


def nwu2enu(nwu):
    """ Convert vector in NWU to ENU coordinate system

    Parameters
    ----------
    nwu (np.array or list of size 3)

    Returns
    -------
    enu (np.array or list of size 3)

    """
    # NWU frame:  (x - forward, y - left, z - up)
    # ENU frame:  (x - right, y - forward, z - up)
    enu = [0, 0, 0]
    enu[0] = -nwu[1]
    enu[1] = nwu[0]
    enu[2] = nwu[2]
    return enu


def nwu2edn(nwu):
    """ Convert vector in NWU to EDN coordinate system

    Parameters
    ----------
    nwu (np.array or list of size 3)

    Returns
    -------
    edn (np.array or list of size 3)

    """
    # NWU frame:  (x - forward, y - left, z - up)
    # EDN frame:  (x - right, y - down, z - forward)
    edn = [0, 0, 0]
    edn[0] = -nwu[1]
    edn[1] = -nwu[2]
    edn[2] = nwu[0]
    return edn
