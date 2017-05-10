#!/usr/bin/env python3
import unittest

import numpy as np
import matplotlib.pylab as plt
from PIL import Image
from scipy import ndimage

from prototype.vision.common import random_3d_features
from prototype.vision.homography import affine_transformation


class HomographyTest(unittest.TestCase):
    # def test_sandbox(self):
    #     with Image.open("data/empire/empire.jpg") as img:
    #         im1 = np.array(img.convert("L"))
    #
    #         H = np.array([[1.4, 0.05, -100],
    #                       [0.05, 1.5, -100],
    #                       [0, 0, 1]])
    #
    #         im2 = ndimage.affine_transform(
    #             im1,
    #             H[:2, :2],
    #             (H[0, 2], H[1, 2])
    #         )
    #
    #         plt.figure()
    #         plt.gray()
    #         plt.imshow(im2)
    #         plt.show()

    def test_condition_points(self):
        nb_features = 10
        feature_bounds = {
            "x": {"min": -1.0, "max": 1.0},
            "y": {"min": -1.0, "max": 1.0},
            "z": {"min": -1.0, "max": 1.0}
        }
        features = random_3d_features(nb_features, feature_bounds)
        fp = np.array(features).T

        m = np.mean(fp[:2], axis=1)
        maxstd = max(np.std(fp[:2], axis=1)) + 1e-9
        C1 = np.diag([1 / maxstd, 1 / maxstd, 1])
        C1[0][2] = -m[0] / maxstd
        C1[1][2] = -m[1] / maxstd
        fp = np.dot(C1, fp)
