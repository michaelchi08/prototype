import os
from math import pi

import numpy as np

from prototype.utils.utils import nwu2edn
from prototype.utils.data import mat2csv
from prototype.models.two_wheel import two_wheel_2d_model
from prototype.vision.common import camera_intrinsics
from prototype.vision.common import rand3dfeatures
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

        self.landmarks = []
        self.time = []
        self.robot_states = []
        self.observed_landmarks = []

    def generate_features(self):
        """ Setup features """
        features = rand3dfeatures(self.nb_features, self.feature_bounds)
        return features

    def output_robot_state(self, save_dir):
        """ Output robot state

        Args:

            save_dir (str): Path to save output

        """
        # Setup state file
        header = ["time_step", "x", "y", "theta"]
        state_file = open(os.path.join(save_dir, "state.dat"), "w")
        state_file.write(",".join(header) + "\n")

        # Write state file
        for i in range(len(self.time)):
            t = self.time[i]
            x = self.robot_states[i].ravel().tolist()

            state_file.write(str(t) + ",")
            state_file.write(str(x[0]) + ",")
            state_file.write(str(x[1]) + ",")
            state_file.write(str(x[2]) + "\n")

        # Clean up
        state_file.close()

    def output_observed(self, save_dir):
        """ Output observed features

        Args:

            save_dir (str): Path to save output

        """
        # Setup
        index_file = open(os.path.join(save_dir, "index.dat"), "w")

        # Output observed landmarks
        for i in range(len(self.time)):
            # Setup output file
            output_path = save_dir + "/observed_" + str(i) + ".dat"
            index_file.write(output_path + '\n')
            obs_file = open(output_path, "w")

            # Data
            t = self.time[i]
            x = self.robot_states[i].ravel().tolist()
            observed = self.observed_landmarks[i]

            # Output time, robot state, and number of observed features
            obs_file.write(str(t) + '\n')
            obs_file.write(','.join(map(str, x)) + '\n')
            obs_file.write(str(len(observed)) + '\n')

            # Output observed landmarks
            for obs in self.observed_landmarks[i]:
                img_pt, landmark_id = obs

                # Convert to string
                img_pt = ','.join(map(str, img_pt[0:2]))
                landmark_id = str(landmark_id)

                # Write to file
                obs_file.write(img_pt + '\n')
                obs_file.write(landmark_id + '\n')

            # Close observed file
            obs_file.close()

        # Close index file
        index_file.close()

    def output_features(self, save_dir):
        """ Output features

        Args:

            save_dir (str): Path to save output

        """
        mat2csv(os.path.join(save_dir, "landmarks.dat"), self.landmarks)

    def calculate_circle_angular_velocity(self, r, v):
        """ Calculate target circle angular velocity given a desired circle
        radius r and velocity v

        Args:

            r (float): Desired circle radius
            v (float): Desired trajectory velocity

        Returns:

            Target angular velocity to complete a circle of radius r and
            velocity v

        """
        dist = 2 * pi * r
        time = dist / v
        return (2 * pi) / time

    def simulate_test_data(self):
        """ Simulate test data """
        # Initialize states
        dt = 0.01
        time = 0.0
        x = np.array([0, 0, 0]).reshape(3, 1)
        w = self.calculate_circle_angular_velocity(0.5, 1.0)
        u = np.array([1.0, w]).reshape(2, 1)
        self.landmarks = self.generate_features()

        # Simulate two wheel robot
        for i in range(300):
            # Update state
            x = two_wheel_2d_model(x, u, dt)

            # Convert both euler angles and translation from NWU to EDN
            rpy = nwu2edn([0.0, 0.0, x[2]])
            t = nwu2edn([x[0], x[1], 0.0])

            # Check landmark
            observed = self.camera.check_landmarks(dt, self.landmarks, rpy, t)
            if observed is not None:
                self.observed_landmarks.append(observed)
                self.robot_states.append(x)
                self.time.append(time)

            # Update
            time += dt

    def generate_test_data(self, save_dir):
        """ Generate test data

        Args:

            save_dir (str): Path to save output

        """
        # mkdir calibration directory
        os.mkdir(save_dir)

        # Simulate test data
        self.simulate_test_data()

        # Output landmarks and robot state
        self.output_features(save_dir)
        self.output_robot_state(save_dir)
        self.output_observed(save_dir)
