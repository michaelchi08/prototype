import numpy as np
from numpy import dot

from prototype.utils.utils import quat2rot


def reprojection_error(K, image_point, rotation, translation, world_point):
    """ Reprojection Error

    Args:

        K (np.array of size 3x3): Camera intrinsics
        image_point (np.array of size 3x1): Image point
        rotation (np.array of size 4x1): Quaternion(w, x, y, z)
        translation (np.array of size 3x3): Translation
        world_point (np.array of size 3x1): World point

    Returns:

        Reprojection error (float)

    """
    # convert quaterion to rotation matrix R
    R = quat2rot(rotation)
    # print(R)

    # project 3D world point to image plane
    est_homo = dot(K, dot(R, (world_point - translation)))

    # normalize projected image point
    est_pt = np.array([0.0, 0.0])
    est_pt[0] = est_homo[0] / est_homo[2]
    est_pt[1] = est_homo[1] / est_homo[2]

    # calculate residual error
    residual = np.absolute(image_point - est_pt)
    # print("residual: ", residual)

    return residual
