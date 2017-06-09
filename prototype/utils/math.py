from math import pi
from math import cos
from math import sin
from math import atan2
from math import asin
from math import fmod

import numpy as np


def deg2rad(d):
    """Convert degrees to radians"""
    return d * (pi / 180.0)


def rad2deg(r):
    """Convert radians to degrees"""
    return r * (180.0 / pi)


def rotx(theta):
    """Rotation matrix around x-axis"""
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, cos(theta), sin(theta)],
                     [0.0, -sin(theta), cos(theta)]])


def roty(theta):
    """Rotation matrix around y-axis"""
    return np.array([[cos(theta), 0.0, -sin(theta)],
                     [0.0, 1.0, 0.0],
                     [sin(theta), 0.0, cos(theta)]])


def rotz(theta):
    """Rotation matrix around z-axis"""
    return np.array([[cos(theta), sin(theta), 0.0],
                     [-sin(theta), cos(theta), 0.0],
                     [0.0, 0.0, 1.0]])


def euler123(phi, theta, psi):
    R11 = cos(theta) * cos(psi)
    R12 = cos(theta) * sin(psi)
    R13 = -sin(theta)

    R21 = sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi)
    R22 = sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi)
    R23 = sin(phi) * cos(theta)

    R31 = cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)
    R32 = cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi)
    R33 = cos(phi) * cos(theta)

    return np.array([[R11, R12, R13],
                     [R21, R22, R23],
                     [R31, R32, R33]])


def euler321(phi, theta, psi):
    R11 = cos(theta) * cos(psi)
    R12 = sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi)
    R13 = cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)

    R21 = cos(theta) * sin(psi)
    R22 = sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi)
    R23 = cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi)

    R31 = -sin(theta)
    R32 = sin(phi) * cos(theta)
    R33 = cos(phi) * cos(theta)

    return np.array([[R11, R12, R13],
                     [R21, R22, R23],
                     [R31, R32, R33]])


def euler2rot(euler, euler_seq):
    phi = euler[0]
    theta = euler[1]
    psi = euler[2]

    if euler_seq == 321:
        # euler 3-2-1
        R11 = cos(theta) * cos(psi)
        R12 = sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi)
        R13 = cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)

        R21 = cos(theta) * sin(psi)
        R22 = sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi)
        R23 = cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi)

        R31 = -sin(theta)
        R32 = sin(phi) * cos(theta)
        R33 = cos(phi) * cos(theta)

    elif euler_seq == 123:
        # euler 1-2-3
        R11 = cos(theta) * cos(psi)
        R12 = cos(theta) * sin(psi)
        R13 = -sin(theta)

        R21 = sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi)
        R22 = sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi)
        R23 = sin(phi) * cos(theta)

        R31 = cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)
        R32 = cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi)
        R33 = cos(phi) * cos(theta)

    return np.array([[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]])


def euler2quat(euler, euler_seq):
    alpha, beta, gamma = euler
    c1 = cos(alpha / 2.0)
    c2 = cos(beta / 2.0)
    c3 = cos(gamma / 2.0)
    s1 = sin(alpha / 2.0)
    s2 = sin(beta / 2.0)
    s3 = sin(gamma / 2.0)

    if euler_seq == 123:
        # euler 1-2-3 to quaternion
        w = c1 * c2 * c3 - s1 * s2 * s3
        x = s1 * c2 * c3 + c1 * s2 * s3
        y = c1 * s2 * c3 - s1 * c2 * s3
        z = c1 * c2 * s3 + s1 * s2 * c3
        return [w, x, y, z]

    elif euler_seq == 321:
        # euler 3-2-1 to quaternion
        w = c1 * c2 * c3 + s1 * s2 * s3
        x = s1 * c2 * c3 - c1 * s2 * s3
        y = c1 * s2 * c3 + s1 * c2 * s3
        z = c1 * c2 * s3 - s1 * s2 * c3
        return [w, x, y, z]

    else:
        error_msg = "Error! Invalid euler sequence [%s]" % str(euler_seq)
        raise RuntimeError(error_msg)


def quat2euler(q, euler_seq):
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    qw2 = pow(qw, 2)
    qx2 = pow(qx, 2)
    qy2 = pow(qy, 2)
    qz2 = pow(qz, 2)

    if euler_seq == 123:
        phi = atan2(2 * (qz * qw - qx * qy), (qw2 + qx2 - qy2 - qz2))
        theta = asin(2 * (qx * qz + qy * qw))
        psi = atan2(2 * (qx * qw - qy * qz), (qw2 - qx2 - qy2 + qz2))

    elif euler_seq == 321:
        phi = atan2(2 * (qx * qw + qz * qy), (qw2 - qx2 - qy2 + qz2))
        theta = asin(2 * (qy * qw - qx * qz))
        psi = atan2(2 * (qx * qy + qz * qw), (qw2 + qx2 - qy2 - qz2))

    return [phi, theta, psi]


def quat2rot(q):
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    # qw2 = pow(qw, 2)
    qx2 = pow(qx, 2)
    qy2 = pow(qy, 2)
    qz2 = pow(qz, 2)

    # inhomogeneous form
    R11 = 1 - 2 * qy2 - 2 * qz2
    R12 = 2 * qx * qy + 2 * qz * qw
    R13 = 2 * qx * qz - 2 * qy * qw

    R21 = 2 * qx * qy - 2 * qz * qw
    R22 = 1 - 2 * qx2 - 2 * qz2
    R23 = 2 * qy * qz + 2 * qx * qw

    R31 = 2 * qx * qz + 2 * qy * qw
    R32 = 2 * qy * qz - 2 * qx * qw
    R33 = 1 - 2 * qx2 - 2 * qy2

    # # homogeneous form
    # R11 = qx2 + qx2 - qy2 - qz2
    # R12 = 2 * (qx * qy - qw * qz)
    # R13 = 2 * (qw * qy + qx * qz)

    # R21 = 2 * (qx * qy + qw * qz)
    # R22 = qw2 - qx2 + qy2 - qz2
    # R23 = 2 * (qy * qz - qw * qx)

    # R31 = 2 * (qx * qz - qw * qy)
    # R32 = 2 * (qw * qx + qy * qz)
    # R33 = qw2 - qx2 - qy2 + qz2

    return np.array([[R11, R12, R13],
                     [R21, R22, R23],
                     [R31, R32, R33]])


def enu2nwu(enu):
    # ENU frame:  (x - right, y - forward, z - up)
    # NWU frame:  (x - forward, y - left, z - up)
    nwu = [0, 0, 0]
    nwu[0] = enu[1]
    nwu[1] = -enu[0]
    nwu[2] = enu[2]
    return nwu


def cf2nwu(cf):
    # camera frame:  (x - right, y - down, z - forward)
    # NWU frame:  (x - forward, y - left, z - up)
    nwu = [0, 0, 0]
    nwu[0] = cf[2]
    nwu[1] = -cf[0]
    nwu[2] = -cf[1]
    return nwu


def cf2enu(cf):
    # camera frame:  (x - right, y - down, z - forward)
    # ENU frame:  (x - right, y - forward, z - up)
    enu = [0, 0, 0]
    enu[0] = cf[0]
    enu[1] = cf[2]
    enu[2] = -cf[1]
    return enu


def ned2enu(ned):
    # NED frame:  (x - forward, y - right, z - down)
    # ENU frame:  (x - right, y - forward, z - up)
    enu = [0, 0, 0]
    enu[0] = ned[1]
    enu[1] = ned[0]
    enu[2] = -ned[2]
    return enu


def nwu2enu(nwu):
    # NWU frame:  (x - forward, y - left, z - up)
    # ENU frame:  (x - right, y - forward, z - up)
    enu = [0, 0, 0]
    enu[0] = -nwu[1]
    enu[1] = nwu[0]
    enu[2] = nwu[2]
    return enu


def nwu2edn(nwu):
    # NWU frame:  (x - forward, y - left, z - up)
    # EDN frame:  (x - right, y - down, z - forward)
    edn = [0, 0, 0]
    edn[0] = -nwu[1]
    edn[1] = -nwu[2]
    edn[2] = nwu[0]
    return edn


def wrap180(euler_angle):
    return fmod((euler_angle + 180.0), 360.0) - 180.0


def wrap360(euler_angle):
    if euler_angle > 0.0:
        return fmod(euler_angle, 360.0)
    else:
        euler_angle += 360.0
        return fmod(euler_angle, 360.0)
