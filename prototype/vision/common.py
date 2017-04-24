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


def camera_intrinsics(fx, fy, cx, cy):
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ])
    return K


def random_3d_features(nb_features, feature_bounds):
    features = []

    for i in range(nb_features):
        x = randf(feature_bounds["x"]["min"], feature_bounds["x"]["max"])
        y = randf(feature_bounds["y"]["min"], feature_bounds["y"]["max"])
        z = randf(feature_bounds["z"]["min"], feature_bounds["z"]["max"])
        features.append([x, y, z])

    return features
