from math import tan
from random import uniform as randf

import scipy.linalg
import numpy as np

from prototype.utils.utils import deg2rad


def normalize(points):
    """ Normalize a collection of `points` in homogeneous coordinates so that
    the last row equals 1. """
    for row in points:
        row /= points[-1]

    return points


def convert2homogeneous(points):
    """ Convert a set of points (dim * n array) to homogeneous coordinates
    where `points` is a numpy array matrix. Returns points in homogeneous
    coordinates. """
    return np.vstack((points, np.ones((1, points.shape[1]))))


def focal_length(image_width, image_height, fov):
    """ Calculate focal length in the x and y axis from:
    - image width
    - image height
    - field of view
    """
    fx = (image_width / 2.0) / tan(deg2rad(fov) / 2.0)
    fy = (image_height / 2.0) / tan(deg2rad(fov) / 2.0)
    return (fx, fy)


def projection_matrix(K, R, t):
    """ Construct projection matrix

    Args:

        K (np.array 3x3 matrix): Camera intrinsics matrix
        R (np.array 3x3 matrix): Camera rotation matrix
        t (np.array 3x1 vector): Camera translation vector

    Returns:

        Projection Matrix (3 x 4 matrix)

    """
    extrinsics = np.array([[R[0, 0], R[0, 1], R[0, 2], t[0]],
                           [R[1, 0], R[1, 1], R[1, 2], t[1]],
                           [R[2, 0], R[2, 1], R[2, 2], t[2]]])
    P = np.dot(K, extrinsics)
    return P


def factor_projection_matrix(P):
    """ Extract camera intrinsics, rotation matrix and translation vector

    Args:

        P (np.array of size 3 x 4): Projection Matrix

    Returns:

        K (np.array 3x3 matrix): Camera intrinsics matrix
        R (np.array 3x3 matrix): Camera rotation matrix
        t (np.array 3x1 vector): Camera translation vector

    """
    K, R = scipy.linalg.rq(P[:, :3])

    # RQ-factorization is not unique, there is a sign ambiguity in the
    # factorization. Since we need the rotation matrix R to have positive
    # determinant (otherwise coordinate axis can get flipped) we can add a
    # transform T to change the sign when needed
    T = np.diag(np.sign(np.diag(K)))
    if np.linalg.det(T) < 0:
        T[1, 1] *= -1

    K = np.dot(K, T)
    R = np.dot(T, R)  # T is its own inverse
    t = np.dot(np.linalg.inv(K), P[:, 3])
    return K, R, t


def camera_center(P):
    """ Extract camera center from projection matrix P

    Args:

        P (np.array of size 3 x 4): Projection Matrix

    """
    K, R, t = factor_projection_matrix(P)
    return np.dot(R.T, t)


def camera_intrinsics(fx, fy, cx, cy):
    """ Construct camera intrinsics matrix K

    Args:

        fx (float): Focal length in x-axis
        fy (float): Focal length in y-axis
        cx (float): Principle point in x-axis
        cy (float): Principle point in y-axis

    Returns:

        Camera intrinsics matrix as a 3x3 matrix

    """
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]])
    return K


def rand3dfeatures(nb_features, feature_bounds):
    """ Generate random 3D features

    Args:

        bounds (dict): 3D feature bounds, for example
                       bounds = {
                         "x": {"min": -1.0, "max": 1.0},
                         "y": {"min": -1.0, "max": 1.0},
                         "z": {"min": -1.0, "max": 1.0}
                       }

        nb_features (int): number of 3D features to generate

    Returns:

        features (list): list of 3D features

    """
    features = []

    for i in range(nb_features):
        x = randf(feature_bounds["x"]["min"], feature_bounds["x"]["max"])
        y = randf(feature_bounds["y"]["min"], feature_bounds["y"]["max"])
        z = randf(feature_bounds["z"]["min"], feature_bounds["z"]["max"])
        features.append([x, y, z])

    return features
