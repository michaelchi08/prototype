#!/usr/bin/env python3
import numpy as np
from numpy import dot

from prototype.utils.math import nwu2edn
from prototype.utils.math import euler2rot
from prototype.vision.common import projection_matrix


class PinHoleCameraModel(object):
    def __init__(self, image_width, image_height, hz, K):
        self.image_width = image_width
        self.image_height = image_height
        self.hz = hz
        self.K = K

        self.frame = 0
        self.dt = 0.0

    def update(self, dt):
        """Update camera"""
        self.dt += dt

        if self.dt > (1.0 / self.hz):
            self.dt = 0.0
            self.frame += 1
            return True

        return False

    def project(self, X, R, t):
        """ Project 3D point to image plane """
        P = projection_matrix(self.K, R, dot(-R, t))
        x = dot(P, X)
        for i in range(3):
            x[i] /= x[2]
        return x

    def check_features(self, dt, features, rpy, t):
        """ Check whether features are observable by camera """
        observed = []

        # pre-check
        if self.update(dt) == False:
            return observed

        # rotation matrix - convert from nwu to edn then to rotation matrix R
        rpy_edn = nwu2edn(rpy)
        R = euler2rot(rpy_edn, 123)

        # translation - convert translation from nwu to edn
        t_edn = nwu2edn(t)

        # projection matrix
        P = projection_matrix(self.K, R, dot(-R, t_edn))

        # check which features in 3d are observable from camera
        for i in range(len(features)):
            # convert feature in NWU to EDN coordinate system
            f3d = features[i]
            f3d_edn = [0, 0, 0, 0]
            f3d_edn[0] = -f3d[1]
            f3d_edn[1] = -f3d[2]
            f3d_edn[2] = f3d[0]
            f3d_edn[3] = 1.0
            f3d_edn = np.array(f3d_edn)

            # project 3D world point to 2D image plane
            f2d = dot(P, f3d_edn)

            # check to see if feature is valid and infront of camera
            if f2d[2] < 1.0:
                continue  # feature is not infront of camera skip

            # normalize pixels
            f2d[0] = f2d[0] / f2d[2]
            f2d[1] = f2d[1] / f2d[2]
            f2d[2] = f2d[2] / f2d[2]

            # check to see if feature observed is within image plane
            x_ok = (f2d[0] < self.image_width) and (f2d[0] > 0.0)
            y_ok = (f2d[1] < self.image_height) and (f2d[1] > 0.0)
            if x_ok and y_ok:
                observed.append((f2d[0:2], np.array(f3d[0:3])))

        return observed
