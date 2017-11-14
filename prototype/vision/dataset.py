import os
from math import pi

import numpy as np

from prototype.utils.utils import nwu2edn
from prototype.utils.data import mat2csv
from prototype.models.two_wheel import two_wheel_2d_model
from prototype.vision.common import camera_intrinsics
from prototype.vision.common import rand3dfeatures
from prototype.vision.camera_model import PinholeCameraModel


class DatasetGenerator(object):
    """Dataset Generator

    Attributes
    ----------
    camera_model : PinholeCameraModel
        Camera model
    nb_features : int
        Number of features
    feature_bounds : dict
        feature_bounds = {
            "x": {"min": -10.0, "max": 10.0},
            "y": {"min": -10.0, "max": 10.0},
            "z": {"min": -10.0, "max": 10.0}
        }

    """

    def __init__(self):
        K = camera_intrinsics(554.25, 554.25, 320.0, 320.0)
        self.camera_model = PinholeCameraModel(640, 640, K, hz=10)
        self.nb_features = 1000
        self.feature_bounds = {
            "x": {"min": -10.0, "max": 10.0},
            "y": {"min": -10.0, "max": 10.0},
            "z": {"min": -10.0, "max": 10.0}
        }

        self.features = []
        self.time = []
        self.robot_states = []
        self.observed_features = []

    def generate_features(self):
        """Setup features"""
        features = rand3dfeatures(self.nb_features, self.feature_bounds)
        return features

    def output_robot_state(self, save_dir):
        """Output robot state

        Parameters
        ----------
        save_dir : str
            Path to save output

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
        """Output observed features

        Parameters
        ----------
        save_dir : str
            Path to save output

        """
        # Setup
        index_file = open(os.path.join(save_dir, "index.dat"), "w")

        # Output observed features
        for i in range(len(self.time)):
            # Setup output file
            output_path = save_dir + "/observed_" + str(i) + ".dat"
            index_file.write(output_path + '\n')
            obs_file = open(output_path, "w")

            # Data
            t = self.time[i]
            x = self.robot_states[i].ravel().tolist()
            observed = self.observed_features[i]

            # Output time, robot state, and number of observed features
            obs_file.write(str(t) + '\n')
            obs_file.write(','.join(map(str, x)) + '\n')
            obs_file.write(str(len(observed)) + '\n')

            # Output observed features
            for obs in self.observed_features[i]:
                img_pt, feature_id = obs

                # Convert to string
                img_pt = ','.join(map(str, img_pt[0:2]))
                feature_id = str(feature_id)

                # Write to file
                obs_file.write(img_pt + '\n')
                obs_file.write(feature_id + '\n')

            # Close observed file
            obs_file.close()

        # Close index file
        index_file.close()

    def output_features(self, save_dir):
        """Output features

        Parameters
        ----------
        save_dir : str
            Path to save output

        """
        mat2csv(os.path.join(save_dir, "features.dat"), self.features)

    def calculate_circle_angular_velocity(self, r, v):
        """Calculate target circle angular velocity given a desired circle
        radius r and velocity v

        Parameters
        ----------
        r : float
            Desired circle radius
        v : float
            Desired trajectory velocity

        Returns
        -------

            Target angular velocity to complete a circle of radius r and
            velocity v

        """
        dist = 2 * pi * r
        time = dist / v
        return (2 * pi) / time

    def simulate_test_data(self):
        """Simulate test data"""
        # Initialize states
        dt = 0.01
        time = 0.0
        x = np.array([0, 0, 0]).reshape(3, 1)
        w = self.calculate_circle_angular_velocity(0.5, 1.0)
        u = np.array([0.1, 0.0]).reshape(2, 1)
        self.features = self.generate_features()

        # Simulate two wheel robot
        for i in range(300):
            # Update state
            x = two_wheel_2d_model(x, u, dt)

            # Convert both euler angles and translation from NWU to EDN
            rpy = nwu2edn([0.0, 0.0, x[2]])
            t = nwu2edn([x[0], x[1], 0.0])

            # Check feature
            observed = self.camera_model.check_features(dt, self.features, rpy, t)
            if observed is not None:
                self.observed_features.append(observed)
                self.robot_states.append(x)
                self.time.append(time)

            # Update
            time += dt

    def generate_test_data(self, save_dir):
        """Generate test data

        Parameters
        ----------
        save_dir : str
            Path to save output

        """
        # mkdir calibration directory
        os.mkdir(save_dir)

        # Simulate test data
        self.simulate_test_data()

        # Output features and robot state
        self.output_features(save_dir)
        self.output_robot_state(save_dir)
        self.output_observed(save_dir)
