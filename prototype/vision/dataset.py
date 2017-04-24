#!/usr/bin/env python3
import os
from math import pi

import numpy as np
from numpy import dot

from prototype.models.two_wheel import two_wheel_2d_model
from prototype.utils.math import nwu2edn
from prototype.utils.math import euler2rot
from prototype.utils.data import mat2csv
from prototype.vision.common import projection_matrix
from prototype.vision.common import camera_intrinsics
from prototype.vision.common import random_3d_features


class SimCamera(object):

    def __init__(self, image_width, image_height, hz, K):
        self.image_width = image_width
        self.image_height = image_height
        self.hz = hz
        self.K = K

        self.frame = 0
        self.dt = 0.0

    def update(self, dt):
        self.dt += dt

        if self.dt > (1.0 / self.hz):
            self.dt = 0.0
            self.frame += 1
            return True

        return False

    def check_features(self, dt, features, rpy, t):
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


class DatasetGenerator(object):
    def __init__(self):
        K = camera_intrinsics(554.25, 554.25, 320.0, 320.0)
        self.camera = SimCamera(640, 640, 10, K)
        self.nb_features = 100
        self.feature_bounds = {
            "x": {"min": -10.0, "max": 10.0},
            "y": {"min": -10.0, "max": 10.0},
            "z": {"min": -10.0, "max": 10.0}
        }

    def setup_state_file(self, save_dir):
        header = ["time_step", "x", "y", "theta"]
        state_file = open(os.path.join(save_dir, "state.dat"), "w")
        state_file.write(",".join(header) + "\n")
        return state_file

    def setup_index_file(self, save_dir):
        index_file = open(os.path.join(save_dir, "index.dat"), "w")
        return index_file

    def setup_features(self, save_dir):
        features = random_3d_features(self.nb_features, self.feature_bounds)
        mat2csv(os.path.join(save_dir, "features.dat"), features)
        return features

    def record_robot_state(self, output_file, x):
        output_file.write(str(x))

    def record_observed_features(self, save_dir, index_file, time, x, observed):
        # setup
        output_path = save_dir + "/observed_" + str(self.camera.frame) + ".dat"
        index_file.write(output_path + '\n')
        outfile = open(output_path, "w")

        # time and number of observed features
        outfile.write(str(time) + '\n')
        outfile.write(','.join(map(str, x)) + '\n')
        outfile.write(str(len(observed)) + '\n')

        # features
        for obs in observed:
            f2d, f3d = obs
            outfile.write(','.join(map(str, f2d[0:2])))  # in image frame
            outfile.write('\n')
            outfile.write(','.join(map(str, f3d[0:3])))  # in world frame
            outfile.write('\n')

        # clean up
        outfile.close()

    def calculate_circle_angular_velocity(self, r, v):
        dist = 2 * pi * r
        time = dist / v
        return (2 * pi) / time

    def generate_test_data(self, save_dir):
        # mkdir calibration directory
        os.mkdir(save_dir)

        # setup
        state_file = self.setup_state_file(save_dir)
        index_file = self.setup_index_file(save_dir)
        features = self.setup_features(save_dir)

        # initialize states
        dt = 0.01
        time = 0.0
        x = [0, 0, 0]
        w = self.calculate_circle_angular_velocity(0.5, 1.0)
        u = [1.0, w]

        # simulate two wheel robot
        for i in range(300):
            # update state
            x = two_wheel_2d_model(x, u, dt)
            time += dt

            # check features
            rpy = [0.0, 0.0, x[2]]
            t = [x[0], x[1], 0.0]
            observed = self.camera.check_features(dt, features, rpy, t)
            if len(observed) > 0:
                self.record_observed_features(save_dir,
                                              index_file,
                                              time,
                                              x,
                                              observed)

            # record state
            self.record_robot_state(state_file, x)

        # clean up
        state_file.close()
        index_file.close()
