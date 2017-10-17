from math import sqrt
from math import cos
from math import sin
from math import acos

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
    return np.array([[qw * px + qz * py - qy * pz + qx * pw],
                     [-qz * px + qw * py + qx * pz + qy * pw],
                     [qy * px - qx * py + qw * pz + qz * pw],
                     [-qx * px - qy * py - qz * pz + qw * pw]])


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


class Quaternion:
    def from_axis_angle(theta, v):
        theta = theta
        v = normalize(v)

        new_quaternion = Quaternion()
        new_quaternion._axis_angle_to_q(theta, v)
        return new_quaternion

    def from_value(value):
        new_quaternion = Quaternion()
        new_quaternion._val = value
        return new_quaternion

    def _axis_angle_to_q(self, theta, v):
        x = v[0]
        y = v[1]
        z = v[2]

        w = cos(theta/2.)
        x = x * sin(theta/2.)
        y = y * sin(theta/2.)
        z = z * sin(theta/2.)

        self._val = np.array([w, x, y, z])

    def _quatmul(self, q2):
        w1, x1, y1, z1 = self._val
        w2, x2, y2, z2 = q2._val
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

        result = Quaternion.from_value(np.array((w, x, y, z)))
        return result

    def _vecmul(self, v):
        q2 = Quaternion.from_value(np.append((0.0), v))
        return (self * q2 * self.get_conjugate())._val[1:]

    def conjugate(self):
        w, x, y, z = self._val
        result = Quaternion.from_value(np.array((w, -x, -y, -z)))
        return result

    def get_axis_angle(self):
        w, v = self._val[0], self._val[1:]
        theta = acos(w) * 2.0

        return theta, normalize(v)

    def tolist(self):
        return self._val.tolist()

    def vector_norm(self):
        w, v = self.get_axis_angle()
        return np.linalg.norm(v)

    def __mul__(self, b):
        if isinstance(b, Quaternion):
            return self._quatmul(b)

        elif isinstance(b, (list, tuple, np.ndarray)):
            if len(b) != 3:
                e = "Input vector has invalid length {}".format(len(b))
                raise Exception(e)

            return self._vecmul(b)

        else:
            e = "Multiplication with unknown type {}".format(type(b))
            raise Exception(e)

    def __repr__(self):
        theta, v = self.get_axis_angle()
        return "((%.6f; %.6f, %.6f, %.6f))" % (theta, v[0], v[1], v[2])
