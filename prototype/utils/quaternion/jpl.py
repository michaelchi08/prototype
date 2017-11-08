from math import sqrt

import numpy as np

from prototype.utils.linalg import skew


def quatnorm(q):
    """ Norm of JPL quaternion

    Args:

        q (np.array - 4x1): Quaternion (x, y, z, w)

    Return:

        Norm of quaternion (float)

    """
    q1, q2, q3, q4 = q.ravel()
    return sqrt(sum(x**2 for x in q))


def quatnormalize(q):
    """ Normalize JPL quaternion

    Args:

        q (np.array - 4x1): Quaternion (x, y, z, w)

    Return:

        Normalized quaternion (np.array - 4x1)

    """
    q1, q2, q3, q4 = q.ravel()
    mag = quatnorm(q)
    q1 = q1 / mag
    q2 = q2 / mag
    q3 = q3 / mag
    q4 = q4 / mag
    return np.array([[q1], [q2], [q3], [q4]])


def quatconj(q):
    """ Conjugate / inverse JPL quaternion

    Source:

        Page 4.

        Trawny, Nikolas, and Stergios I. Roumeliotis. "Indirect Kalman filter
        for 3D attitude estimation." University of Minnesota, Dept. of Comp.
        Sci. & Eng., Tech. Rep 2 (2005): 2005.

    Args:

        q (np.array - 4x1): Quaternion (x, y, z, w)

    Return:

        Quaternion conjugate (np.array - 4x1)

    """
    q1, q2, q3, q4 = q.ravel()
    return np.array([[-q1], [-q2], [-q3], [q4]])


def quatmul(p, q):
    """ Muliply JPL quaternions

    Source:

        Page 3.

        Trawny, Nikolas, and Stergios I. Roumeliotis. "Indirect Kalman filter
        for 3D attitude estimation." University of Minnesota, Dept. of Comp.
        Sci. & Eng., Tech. Rep 2 (2005): 2005.

    Args:

        p (np.array - 4x1): 1st Quaternion (x, y, z, w)
        q (np.array - 4x1): 2nd Quaternion (x, y, z, w)

    Return:

        Product of quaternion multiplication (np.array - 4x1)

    """
    p1, p2, p3, p4 = p.ravel()
    q1, q2, q3, q4 = q.ravel()

    return np.array([[q4 * p1 + q3 * p2 - q2 * p3 + q1 * p4],
                     [-q3 * p1 + q4 * p2 + q1 * p3 + q2 * p4],
                     [q2 * p1 - q1 * p2 + q4 * p3 + q3 * p4],
                     [-q1 * p1 - q2 * p2 - q3 * p3 + q4 * p4]])


def quat2rot(q):
    """ JPL Quaternion to rotation matrix

    Source:

        Page 9.

        Trawny, Nikolas, and Stergios I. Roumeliotis. "Indirect Kalman filter
        for 3D attitude estimation." University of Minnesota, Dept. of Comp.
        Sci. & Eng., Tech. Rep 2 (2005): 2005.

    Args:

        q (np.array - 4x1): 1st Quaternion (x, y, z, w)

    Return:

        Product of quaternion multiplication

    """
    q1, q2, q3, q4 = q.ravel()

    R11 = 1.0 - 2.0 * q2**2.0 - 2.0 * q3**2.0
    R12 = 2.0 * (q1 * q2 + q3 * q4)
    R13 = 2.0 * (q1 * q3 - q2 * q4)

    R21 = 2.0 * (q1 * q2 - q3 * q4)
    R22 = 1.0 - 2.0 * q1**2.0 - 2.0 * q3**2.0
    R23 = 2.0 * (q2 * q3 + q1 * q4)

    R31 = 2.0 * (q1 * q3 + q2 * q4)
    R32 = 2.0 * (q2 * q3 - q1 * q4)
    R33 = 1.0 - 2.0 * q1**2.0 - 2.0 * q2**2.0

    return np.array([[R11, R12, R13],
                     [R21, R22, R23],
                     [R31, R32, R33]])


def Omega(w):
    """ Omega function

    Args:

        w (np.array): Angular velocity

    Returns:

        Differential form of an angular velocity (np.array)

    """
    w = w.reshape((3, 1))
    return np.block([[-skew(w), w], [-w.T, 0.0]])


def quatlcomp(q):
    """ Quaternion left compound

    Source:

        Page 4.

        Trawny, Nikolas, and Stergios I. Roumeliotis. "Indirect Kalman filter
        for 3D attitude estimation." University of Minnesota, Dept. of Comp.
        Sci. & Eng., Tech. Rep 2 (2005): 2005.

    Args:

        q (np.array - 4x1): Quaternion (x, y, z, w)

    """
    q1, q2, q3, q4 = q.ravel()
    vector = np.array([[q1], [q2], [q3]])
    scalar = q4

    return np.block([[scalar * np.eye(3) - skew(vector), vector],
                     [-vector.T, scalar]])
