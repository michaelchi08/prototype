#!/usr/bin/env python3
from math import tan
from random import uniform as randf

import numpy as np
from numpy import dot

from prototype.utils.math import deg2rad


def focal_length(image_width, image_height, fov):
    fx = (image_width / 2.0) / tan(deg2rad(fov) / 2.0)
    fy = (image_height / 2.0) / tan(deg2rad(fov) / 2.0)
    return (fx, fy)


def projection_matrix(K, R, t):
    extrinsics = np.array([[R[0, 0], R[0, 1], R[0, 2], t[0]],
                           [R[1, 0], R[1, 1], R[1, 2], t[1]],
                           [R[2, 0], R[2, 1], R[2, 2], t[2]]])
    P = dot(K, extrinsics)
    return P


def factor_projection_matrix(P):
    K, R = np.linalg.rq(P[:, :3])
    T = np.diag(np.sign(np.diag(K)))

    # RQ-factorization is not unique, there is a sign ambiguity in the
    # factorization. Since we need the rotation matrix R to have positive
    # determinant (otherwise coordinate axis can get flipped) we can add a
    # transform T to change the sign when needed
    if np.linalg.det(T) < 0:
        T[1, 1] *= -1
        K = np.dot(K, T)
        R = np.dot(T, R)  # T is its own inverse
        t = np.dot(np.linalg.inv(K), P[:, 3])

    return K, R, t


def camera_center(P):
    K, R, t = factor_projection_matrix(P)
    return -1 * np.dot(R.T, t)


def camera_intrinsics(fx, fy, cx, cy):
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ])
    return K


def convert2homogeneous(points):
    return np.vstack((points, np.ones((1, points.shape[1]))))


def random_3d_features(nb_features, feature_bounds):
    features = []

    for i in range(nb_features):
        x = randf(feature_bounds["x"]["min"], feature_bounds["x"]["max"])
        y = randf(feature_bounds["y"]["min"], feature_bounds["y"]["max"])
        z = randf(feature_bounds["z"]["min"], feature_bounds["z"]["max"])
        features.append([x, y, z])

    return features
