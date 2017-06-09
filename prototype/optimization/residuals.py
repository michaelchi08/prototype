import numpy as np
from numpy import dot

from prototype.utils.math import quat2rot
from prototype.utils.math import euler2rot
from prototype.vision.common import projection_matrix


def reprojection_error(K, image_point, rotation, translation, world_point):
    # convert quaterion to rotation matrix R
    R = quat2rot(rotation)

    # project 3D world point to image plane
    est_homo = dot(K, dot(R, (world_point - translation)))

    # normalize projected image point
    est_pt = np.array([0.0, 0.0])
    est_pt[0] = est_homo[0] / est_homo[2]
    est_pt[1] = est_homo[1] / est_homo[2]

    # calculate residual error
    residual = np.absolute(image_point - est_pt)

    return residual
