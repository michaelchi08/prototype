from math import sqrt
from math import atan

import numpy as np

def equi_distort(point, D):
    """ Distort 2D point in ideal coordinates using the Equidistant distortion
    model

    Parameters
    ----------
    point : np.array
        2D point in ideal coordinates (not pixel) to be distorted
    D : np.array
        Equidistant distortion coefficients (k1, k2, k3, k4)

    """
    # Calculate distortion
    # -- Decompose distortion coefficients
    k1, k2, k3, k4 = D
    # -- Calculate distortion
    th2 = pow(theta, 2)
    th4 = pow(theta, 4)
    th6 = pow(theta, 6)
    th8 = pow(theta, 8)
    theta_d = theta * (1 + k1 * th2 + k2 * th4 + k3 * th6 + k4 * th8)
    # -- Distort point
    x, y = point
    x_dash = (theta_d / r) * x
    y_dash = (theta_d / r) * y

    return [x_dash, y_dash]


def project_pinhole_equi(X_c, K, D):
    """ Project 3D point in the camera frame onto the image plane using the
    pinhole-equi model.

    This function uses the same distortion model as OpenCV's:

        https://docs.opencv.org/3.2.0/db/d58/group__calib3d__fisheye.html

    Parameters
    ----------
    X_c : np.array
        3D position of a point relative to camera frame
    K : np.array
        Camera intrinsics matrix K (3x3)
    D : np.array
        Equidistant distortion coefficients (k1, k2, k3, k4)

    """
    # Calculate theta (Angle from the principle axis to the 3D points)
    # -- Normalize 3D points
    a = X_c[0] / X_c[2]
    b = X_c[1] / X_c[2]
    # -- Calculate theta for each 3D point
    r = sqrt(pow(a, 2) + pow(b, 2))
    theta = atan(r)

    if np.array_equal(D, np.zeros((4,))) is False:
        # Calculate distortion
        # -- Decompose distortion coefficients
        k1, k2, k3, k4 = D
        # -- Calculate distortion
        th2 = pow(theta, 2)
        th4 = pow(theta, 4)
        th6 = pow(theta, 6)
        th8 = pow(theta, 8)
        theta_d = theta * (1 + k1 * th2 + k2 * th4 + k3 * th6 + k4 * th8)
        # -- Distort point
        x_dash = (theta_d / r) * a
        y_dash = (theta_d / r) * b
    else:
        x_dash = a
        y_dash = b

    # Project distorted points to pixel coordinates
    # -- Decompose camera intrinsics
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    # -- Project distorted points to pixel coordinates using pinhole model
    u = fx * x_dash + cx
    v = fy * y_dash + cy
    pixel = [u, v]

    return pixel
