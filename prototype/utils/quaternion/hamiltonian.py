from math import sqrt
from math import cos
from math import sin
from math import asin
from math import atan2

import numpy as np


def quatnorm(q):
    """Norm of quaternion

    Parameters
    ----------
    q :
        np

    Returns
    -------
    type
        Norm of quaternion (float)

    """
    q1, q2, q3, q4 = q.ravel()
    return sqrt(sum(x**2 for x in q))


def quatnormalize(q):
    """Normalize quaternion

    Parameters
    ----------
    q :
        np

    Returns
    -------
    type
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
    """Conjugate / inverse of a quaternion

    Parameters
    ----------
    q :
        np

    Returns
    -------
    type
        Quaternion conjugate (np.array - 4x1)

    """
    qw, qx, qy, qz = q.ravel()
    return np.array([[qw], [-qx], [-qy], [-qz]])


def quatmul(p, q):
    """Multiply Hamiltonian quaternion

    Parameters
    ----------
    p : np.array
        1st Quaternion (w, x, y, z)
    q : np.array
        2nd Quaternion (w, x, y, z)

    Returns
    -------

        Product of quaternion multiplication (np.array - 4x1)

    """
    pw, px, py, pz = q
    qw, qx, qy, qz = q
    return np.array([[pw * qw - px * qx - py * qy - pz * qz],
                     [pw * qx + px * qw + py * qz - pz * qy],
                     [pw * qy - px * qz + py * qw + pz * qx],
                     [pw * qz + px * qy - py * qx + pz * qw]])


def quatangle(angle):
    """Quaternion from angle

    Parameters
    ----------
    angle :


    Returns
    -------

    """
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
    """Quaternion to rotation matrix

    Parameters
    ----------
    q : np.array or list of size 4
        Quaternion (w, x, y, z)

    Returns
    -------

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
    """Quaternion to Euler

    Parameters
    ----------
    q : np.array
        Quaternion (w, x, y, z)

    euler_seq : int
        Euler sequence

    Returns
    -------
    euler : np.array
        Euler angles

    """
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


def euler2quat(euler, euler_seq):
    """Convert euler to quaternion
    This function assumes we are performing an intrinsic rotation.

    Source:

        Kuipers, Jack B. Quaternions and Rotation Sequences: A Primer with
        Applications to Orbits, Aerospace, and Virtual Reality. Princeton, N.J:
        Princeton University Press, 1999. Print.

        Page 167.

    Parameters
    ----------
    euler : np.array
        Euler angle (roll, pitch, yaw)
    euler_seq : float
        Euler rotation sequence

    Returns
    -------

        Quaternion (np.array)

    """
    if euler_seq == 321:
        phi, theta, psi = euler
        c_phi = cos(phi / 2.0)
        c_theta = cos(theta / 2.0)
        c_psi = cos(psi / 2.0)
        s_phi = sin(phi / 2.0)
        s_theta = sin(theta / 2.0)
        s_psi = sin(psi / 2.0)

        qw = c_psi * c_theta * c_phi + s_psi * s_theta * s_phi
        qx = c_psi * c_theta * s_psi - s_psi * s_theta * c_phi
        qy = c_psi * s_theta * c_phi + s_psi * c_theta * s_phi
        qz = s_psi * c_theta * c_phi - c_psi * s_theta * s_phi

        q = np.array([qw, qx, qy, qz])
        return quatnormalize(q)

    else:
        err_msg = "Error! Unsupported euler sequence [%s]" % str(euler_seq)
        raise RuntimeError(err_msg)
