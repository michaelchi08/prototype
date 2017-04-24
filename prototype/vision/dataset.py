#!/usr/bin/env python3
import os
from math import pi
from random import uniform as randf

from numpy import dot

from prototype.models.two_wheel import two_wheel_2d_model
from prototype.utils.math import nwu2edn
from prototype.utils.math import euler2rot
from prototype.utils.data import mat2csv
from prototype.vision.utils import projection_matrix
from prototype.vision.utils import camera_intrinsics


class SimCamera(object):

    def __init__(self, image_width, image_height, hz, K):
        self.image_width = image_width
        self.image_height = image_height
        self.hz = hz
        self.K = K
        self.frame = 0

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
            return True

        # rotation matrix - convert from nwu to edn then to rotation matrix R
        rpy_edn = nwu2edn(rpy)
        R = euler2rot(rpy_edn, 123)

        # translation - convert translation from nwu to edn
        t_edn = nwu2edn(t)

        # projection matrix
        P = projection_matrix(self.K, R, dot(-R, t_edn))

        # check which features in 3d are observable from camera
        for i in range(features.shape[1]):
            # convert feature in NWU to EDN coordinate system
            f_3d = features[0:4, i]
            f_3d_edn = [0, 0, 0, 0]
            f_3d_edn[0] = -f_3d[1]
            f_3d_edn[1] = -f_3d[2]
            f_3d_edn[2] = f_3d[0]
            f_3d_edn[3] = 1.0

            # project 3D world point to 2D image plane
            f_2d = dot(P, f_3d_edn)

            # check to see if feature is valid and infront of camera
            if f_2d[2] >= 1.0:
                # normalize pixels
                f_2d[0] = f_2d[0] / f_2d[2]
                f_2d[1] = f_2d[1] / f_2d[2]
                f_2d[2] = f_2d[2] / f_2d[2]

                # check to see if feature observed is within image plane
                if (f_2d[0] < self.image_width) and (f_2d[0] > 0.0):
                    if (f_2d[1] < self.image_height) and (f_2d[1] > 0):
                        observed.append((f_2d[0:2], f_3d[0:3]))

        return observed


class DatasetGenerator(object):
    def __init__(self):
        K = camera_intrinsics(554.25, 554.25, 320.0, 320.0)
        self.camera = SimCamera(640, 640, 10, K)
        self.nb_features = 100
        self.bounds = {
            "x": {"min": -1.0, "max": 1.0},
            "y": {"min": -1.0, "max": 1.0},
            "z": {"min": -1.0, "max": 1.0}
        }

    def prep_header(self, output_file):
        header = ["time_step", "x", "y", "theta"]
        output_file.write(header.join(",") + "\n")

    def record_observation(self, output_file, x):
        output_file.write(x.join(",") + "\n")

    def calculate_circle_angular_velocity(self, r, v):
        dist = 2 * pi * r
        time = dist / v
        return (2 * pi) / time

    def generate_random_3d_features(self):
        features = []

        # generate random 3d features
        for i in range(self.nb_features):
            point = [0, 0, 0, 0]
            point[0] = randf(self.bounds["x"]["min"], self.bounds["x"]["max"])
            point[1] = randf(self.bounds["y"]["min"], self.bounds["y"]["max"])
            point[2] = randf(self.bounds["z"]["min"], self.bounds["z"]["max"])
            point[3] = 1.0
            features.append(point)

    def record_3d_features(self, output_path, features):
        mat2csv(output_path, features)

    def record_observed_features(time, x, output_path, observed):
        # setup
        outfile = open(output_path, "w")

        # time and number of observed features
        outfile.write(time)
        outfile.write(len(observed))
        outfile.write(x.join(","))

        # features
        for obs in observed:
            f_2d, f_3d = obs
            outfile.write(f_2d[0:2])  # feature in image frame
            outfile.write(f_3d[0:3])  # feature in world frame

        # clean up
        outfile.close()

    def generate_test_data(self, save_path):
        # mkdir calibration directory
        os.mkdir(save_path)

        # setup
        output_file = open(save_path + "/state.dat", "w")
        self.prep_header(output_file)

        index_file = open(save_path + "/index.dat", "w")

        features = self.generateRandom3DFeatures()
        self.record3DFeatures("/tmp/test/features.dat", features)

        # initialize states
        dt = 0.01
        time = 0.0

        x = [0, 0, 0]

        w = self.calculate_circle_angular_velocity(0.5, 1.0)
        u = [1.0, w]

        for i in range(300):
            # update state
            x = two_wheel_2d_model(x, u, dt)
            time += dt

            # check features
            rpy = [0.0, 0.0, x[2]]
            t = [x[0], x[1], 0.0]
            observed = self.camera.check_features(dt, features, rpy, t)
            if len(observed) > 0:
                observed_file = "/observed_" + str(self.camera.frame) + ".dat"
                self.record_observed_features(time,
                                              x,
                                              save_path + observed_file,
                                              observed)

            # record state
            self.record_observation(output_file, x)

        # clean up
        output_file.close()
        index_file.close()
