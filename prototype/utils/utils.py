from math import pi
from math import cos
from math import sin
from math import atan2
from math import asin
from math import fmod
from math import sqrt

import numpy as np


def deg2rad(d):
    """ Convert degrees to radians

    Args:

        d (float): degrees

    Returns:

        Radians

    """
    return d * (pi / 180.0)


def rad2deg(r):
    """ Convert radians to degrees

    Args:

        r (float): Radians

    Returns:

        Degrees

    """
    return r * (180.0 / pi)


def wrap180(euler_angle):
    """ Wrap angle to 180

    Args:

        euler_angle (float): Euler angle

    Returns:

        Wrapped angle

    """
    return fmod((euler_angle + 180.0), 360.0) - 180.0


def wrap360(euler_angle):
    """ Wrap angle to 360

    Args:

        euler_angle (float): Euler angle

    Returns:

        Wrapped angle

    """
    if euler_angle > 0.0:
        return fmod(euler_angle, 360.0)
    else:
        euler_angle += 360.0
        return fmod(euler_angle, 360.0)


def quatnormalize(q):
    """ Normalize quaternion

    Args:

        q (np.array or list of size 4)

    Returns:

        Normalized quaternion

    """
    qw, qx, qy, qz = q
    qw2 = pow(qw, 2)
    qx2 = pow(qx, 2)
    qy2 = pow(qy, 2)
    qz2 = pow(qz, 2)

    mag = sqrt(qw2 + qx2 + qy2 + qz2)
    q[0] = q[0] / mag
    q[1] = q[1] / mag
    q[2] = q[2] / mag
    q[3] = q[3] / mag

    return q


def rotx(theta):
    """ Rotation matrix around x-axis (counter-clockwise)

    Args:

        theta (float): Rotation around x in radians

    Returns:

        3 x 3 Rotation matrix

    """
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, cos(theta), -sin(theta)],
                     [0.0, sin(theta), cos(theta)]])


def roty(theta):
    """ Rotation matrix around y-axis (counter-clockwise)

    Args:

        theta (float): Rotation around y in radians

    Returns:

        3 x 3 Rotation matrix

    """
    return np.array([[cos(theta), 0.0, sin(theta)],
                     [0.0, 1.0, 0.0],
                     [-sin(theta), 0.0, cos(theta)]])


def rotz(theta):
    """ Rotation matrix around z-axis (counter-clockwise)

    Args:

        theta (float): Rotation around y in radians

    Returns:

        3 x 3 Rotation matrix

    """
    return np.array([[cos(theta), -sin(theta), 0.0],
                     [sin(theta), cos(theta), 0.0],
                     [0.0, 0.0, 1.0]])


def euler2rot(euler, euler_seq):
    """ Convert euler to rotation matrix R
    This function assumes we are performing an extrinsic rotation.

    Source:
        Euler Angles, Quaternions and Transformation Matrices
        https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf
    """
    if euler_seq == 123:
        t1, t2, t3 = euler

        R11 = cos(t2) * cos(t3)
        R12 = -cos(t2) * sin(t3)
        R13 = sin(t2)

        R21 = sin(t1) * sin(t2) * cos(t3) + cos(t1) * sin(t3)
        R22 = -sin(t1) * sin(t2) * sin(t3) + cos(t1) * cos(t3)
        R23 = -sin(t1) * cos(t2)

        R31 = -cos(t1) * sin(t2) * cos(t3) + sin(t1) * sin(t3)
        R32 = cos(t1) * sin(t2) * sin(t3) + sin(t1) * cos(t3)
        R33 = cos(t1) * cos(t2)

    elif euler_seq == 321:
        t3, t2, t1 = euler

        R11 = cos(t1) * cos(t2)
        R12 = cos(t1) * sin(t2) * sin(t3) - sin(t1) * cos(t3)
        R13 = cos(t1) * sin(t2) * cos(t3) + sin(t1) * sin(t3)

        R21 = sin(t1) * cos(t2)
        R22 = sin(t1) * sin(t2) * sin(t3) + cos(t1) * cos(t3)
        R23 = sin(t1) * sin(t2) * cos(t3) - cos(t1) * sin(t3)

        R31 = -sin(t2)
        R32 = cos(t2) * sin(t3)
        R33 = cos(t2) * cos(t3)

    else:
        error_msg = "Error! Unsupported euler sequence [%s]" % str(euler_seq)
        raise RuntimeError(error_msg)

    return np.array([[R11, R12, R13],
                     [R21, R22, R23],
                     [R31, R32, R33]])


def euler2quat(euler, euler_seq):
    """ Convert euler to quaternion
    This function assumes we are performing an extrinsic rotation.

    Source:
        Euler Angles, Quaternions and Transformation Matrices
        https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf
    """

    if euler_seq == 123:
        t1, t2, t3 = euler
        c1 = cos(t1 / 2.0)
        c2 = cos(t2 / 2.0)
        c3 = cos(t3 / 2.0)
        s1 = sin(t1 / 2.0)
        s2 = sin(t2 / 2.0)
        s3 = sin(t3 / 2.0)

        w = -s1 * s2 * s3 + c1 * c2 * c3
        x = s1 * c2 * c3 + s2 * s3 * c1
        y = -s1 * s3 * c2 + s2 * c1 * c3
        z = s1 * s2 * c3 + s3 * c1 * c2
        return quatnormalize([w, x, y, z])

    elif euler_seq == 321:
        t3, t2, t1 = euler
        c1 = cos(t1 / 2.0)
        c2 = cos(t2 / 2.0)
        c3 = cos(t3 / 2.0)
        s1 = sin(t1 / 2.0)
        s2 = sin(t2 / 2.0)
        s3 = sin(t3 / 2.0)

        w = s1 * s2 * s3 + c1 * c2 * c3
        x = -s1 * s2 * c3 + s3 * c1 * c2
        y = s1 * s3 * c2 + s2 * c1 * c3
        z = s1 * c2 * c3 - s1 * s3 * c1
        return quatnormalize([w, x, y, z])

    else:
        error_msg = "Error! Unsupported euler sequence [%s]" % str(euler_seq)
        raise RuntimeError(error_msg)


def quat2euler(q, euler_seq):
    qw, qx, qy, qz = q
    qw2 = pow(qw, 2)
    qx2 = pow(qx, 2)
    qy2 = pow(qy, 2)
    qz2 = pow(qz, 2)

    if euler_seq == 123:
        t1 = atan2(2 * (qz * qw - qx * qy), (qw2 + qx2 - qy2 - qz2))
        t2 = asin(2 * (qx * qz + qy * qw))
        t3 = atan2(2 * (qx * qw - qy * qz), (qw2 - qx2 - qy2 + qz2))
        return np.array([t3, t2, t1])

    elif euler_seq == 321:
        t1 = atan2(2 * (qx * qw + qz * qy), (qw2 - qx2 - qy2 + qz2))
        t2 = asin(2 * (qy * qw - qx * qz))
        t3 = atan2(2 * (qx * qy + qz * qw), (qw2 + qx2 - qy2 - qz2))

        return np.array([t1, t2, t3])

    else:
        error_msg = "Error! Unsupported euler sequence [%s]" % str(euler_seq)
        raise RuntimeError(error_msg)


def quat2rot(q):
    """ Quaternion to rotation matrix

    Args:

        q (np.array or list of size 4): Quaternion (w, x, y, z)

    Returns:

        3 x 3 rotation matrix

    """
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    qw2 = pow(qw, 2)
    qx2 = pow(qx, 2)
    qy2 = pow(qy, 2)
    qz2 = pow(qz, 2)

    # homogeneous form
    R11 = qw2 + qx2 - qy2 - qz2
    R12 = 2 * (qx * qy - qw * qz)
    R13 = 2 * (qx * qz + qw * qy)

    R21 = 2 * (qx * qy + qw * qz)
    R22 = qw2 - qx2 + qy2 - qz2
    R23 = 2 * (qy * qz - qw * qx)

    R31 = 2 * (qx * qz - qw * qy)
    R32 = 2 * (qy * qz + qw * qx)
    R33 = qw2 - qx2 - qy2 + qz2

    return np.array([[R11, R12, R13],
                     [R21, R22, R23],
                     [R31, R32, R33]])


def enu2nwu(enu):
    """ Convert vector in ENU to NWU coordinate system

    Args:

        enu (np.array or list of size 3)

    Returns

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

    Args:

        edn (np.array or list of size 3)

    Returns

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

    Args:

        edn (np.array or list of size 3)

    Returns

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

    Args:

        ned (np.array or list of size 3)

    Returns

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

    Args:

        nwu (np.array or list of size 3)

    Returns

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

    Args:

        nwu (np.array or list of size 3)

    Returns

        edn (np.array or list of size 3)

    """
    # NWU frame:  (x - forward, y - left, z - up)
    # EDN frame:  (x - right, y - down, z - forward)
    edn = [0, 0, 0]
    edn[0] = -nwu[1]
    edn[1] = -nwu[2]
    edn[2] = nwu[0]
    return edn


