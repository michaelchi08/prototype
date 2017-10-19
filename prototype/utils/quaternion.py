from math import sqrt
from math import cos
from math import sin
from math import asin
from math import atan2

import numpy as np


def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = sqrt(mag2)
        v = tuple(n / mag for n in v)
    return np.array(v)


def quatmul(p, q):
    """ Quaternion multiplication

    Args:

        p (np.array): Quaternion (w, x, y, z)
        q (np.array): Quaternion (w, x, y, z)

    Returns:

        Product of p and q as a quaternion (w, x, y, z)

    """
    pw, px, py, pz = q
    qw, qx, qy, qz = q
    return np.array([[pw * qw - px * qx - py * qy - pz * qz],
                     [pw * qx + px * qw + py * qz - pz * qy],
                     [pw * qy - px * qz + py * qw + pz * qx],
                     [pw * qz + px * qy - py * qx + pz * qw]])


def quatnorm(q):
    """ Calculate the norm of a quaternion

    Args:

        q (np.array): Quaternion (w, x, y, z)

    Returns:

        Norm of a quaternion

    """
    qw, qx, qy, qz = q
    qw2 = pow(qw, 2)
    qx2 = pow(qx, 2)
    qy2 = pow(qy, 2)
    qz2 = pow(qz, 2)
    return sqrt(qw2 + qx2 + qy2 + qz2)


def quatnormalize(q):
    """ Normalize quaternion

    Args:

        q (np.array): Quaternion (w, x, y, z)

    Returns:

        Normalized quaternion

    """
    n = quatnorm(q)
    q[0] = q[0] / n
    q[1] = q[1] / n
    q[2] = q[2] / n
    q[3] = q[3] / n

    return q


def quatconj(q):
    """ Conjugate of a quaternion

    Args:

        q (np.array): Quaternion (w, x, y, z)

    Returns:

        Conjugate of a quaternion

    """
    qw, qx, qy, qz = q
    return np.array([[qw], -[qx], -[qy], -[qz]])


def quatinv(q):
    """ Inverse quaternion

    Args:

        q (np.array): Quaternion (w, x, y, z)

    Returns:

        Inverted quaternion

    """
    q_conj = quatconj(q)
    return q_conj / pow(quatnorm(q), 2)


def quatangle(angle):
    """ Quaternion from angle """
    roll, pitch, yaw = angle

    cy = cos(yaw / 2.0)
    sy = sin(yaw / 2.0)
    cr = cos(roll / 2.0)
    sr = sin(roll / 2.0)
    cp = cos(pitch / 2.0)
    sp = sin(pitch / 2.0)

    q = np.zeros(4)
    q[0] = cy * cr * cp + sy * sr * sp
    q[1] = cy * sr * cp - sy * cr * sp
    q[2] = cy * cr * sp + sy * sr * cp
    q[3] = sy * cr * cp - cy * sr * sp

    return q


def quat2rot(q):
    """ Quaternion to rotation matrix

    Args:

        q (np.array or list of size 4): Quaternion (w, x, y, z)

    Returns:

        3 x 3 rotation matrix

    """
    qw, qx, qy, qz = q

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
