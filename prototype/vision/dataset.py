#!/usr/bin/env python3
import os
from math import pi

from prototype.models.two_wheel import two_wheel_2d_model
from prototype.utils.data import mat2csv
from prototype.vision.common import camera_intrinsics
from prototype.vision.common import random_3d_features
from prototype.vision.camera_models import PinholeCameraModel


class DatasetGenerator(object):
    """ Dataset Generator """

    def __init__(self):
        K = camera_intrinsics(554.25, 554.25, 320.0, 320.0)
        self.camera = PinholeCameraModel(640, 640, 10, K)
        self.nb_features = 100
        self.feature_bounds = {
            "x": {"min": -10.0, "max": 10.0},
            "y": {"min": -10.0, "max": 10.0},
            "z": {"min": -10.0, "max": 10.0}
        }

    def setup_state_file(self, save_dir):
        """ Setup state file """
        header = ["time_step", "x", "y", "theta"]
        state_file = open(os.path.join(save_dir, "state.dat"), "w")
        state_file.write(",".join(header) + "\n")
        return state_file

    def setup_index_file(self, save_dir):
        """ Setup index file """
        index_file = open(os.path.join(save_dir, "index.dat"), "w")
        return index_file

    def setup_features(self, save_dir):
        """ Setup features """
        features = random_3d_features(self.nb_features, self.feature_bounds)
        mat2csv(os.path.join(save_dir, "features.dat"), features)
        return features

    def record_robot_state(self, output_file, x):
        """ Record robot state """
        output_file.write(str(x))

    def record_observed_features(self, save_dir, index_file, time, x, observed):
        """ Record observed features """
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
        """ Calculate circle angular velocity """
        dist = 2 * pi * r
        time = dist / v
        return (2 * pi) / time

    def generate_test_data(self, save_dir):
        """ Generate test data """
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
