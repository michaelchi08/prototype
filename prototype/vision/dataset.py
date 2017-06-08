import os
from math import pi

from prototype.utils.math import nwu2edn
from prototype.utils.data import mat2csv
from prototype.models.two_wheel import two_wheel_2d_model
from prototype.vision.common import camera_intrinsics
from prototype.vision.common import rand3dpts
from prototype.vision.camera_models import PinholeCameraModel


class DatasetGenerator(object):
    """ Dataset Generator """

    def __init__(self):
        K = camera_intrinsics(554.25, 554.25, 320.0, 320.0)
        self.camera = PinholeCameraModel(640, 640, 10, K)
        self.nb_landmarks = 100
        self.landmark_bounds = {
            "x": {"min": -10.0, "max": 10.0},
            "y": {"min": -10.0, "max": 10.0},
            "z": {"min": -10.0, "max": 10.0}
        }

        self.landmarks = []
        self.time = []
        self.robot_state = []
        self.observed_landmarks = []

    def generate_landmarks(self):
        """ Setup features """
        features = rand3dpts(self.nb_landmarks, self.landmark_bounds)
        return features

    def output_robot_state(self, save_dir):
        """ Output robot state """

        # setup state file
        header = ["time_step", "x", "y", "theta"]
        state_file = open(os.path.join(save_dir, "state.dat"), "w")
        state_file.write(",".join(header) + "\n")

        # write state file
        for i in range(len(self.time)):
            t = self.time[i]
            x = self.robot_state[i]

            state_file.write(str(t) + ",")
            state_file.write(str(x[0]) + ",")
            state_file.write(str(x[1]) + ",")
            state_file.write(str(x[2]) + "\n")

        # clean up
        state_file.close()

    def output_observed(self, save_dir):
        """ Output observed features """
        # setup
        index_file = open(os.path.join(save_dir, "index.dat"), "w")

        # output observed landmarks
        for i in range(len(self.time)):
            # setup output file
            output_path = save_dir + "/observed_" + str(i) + ".dat"
            index_file.write(output_path + '\n')
            obs_file = open(output_path, "w")

            # data
            t = self.time[i]
            x = self.robot_state[i]
            observed = self.observed_landmarks[i]

            # output time, robot state, and number of observed features
            obs_file.write(str(t) + '\n')
            obs_file.write(','.join(map(str, x)) + '\n')
            obs_file.write(str(len(observed)) + '\n')

            # output observed landmarks
            for obs in self.observed_landmarks[i]:
                img_pt, landmark_id = obs

                # convert to string
                img_pt = ','.join(map(str, img_pt[0:2]))
                landmark_id = str(landmark_id)

                # write to file
                obs_file.write(img_pt + '\n')
                obs_file.write(landmark_id + '\n')

            # close observed file
            obs_file.close()

        # close index file
        index_file.close()

    def output_landmarks(self, save_dir):
        mat2csv(os.path.join(save_dir, "landmarks.dat"), self.landmarks)

    def calculate_circle_angular_velocity(self, r, v):
        """ Calculate circle angular velocity """
        dist = 2 * pi * r
        time = dist / v
        return (2 * pi) / time

    def simulate_test_data(self):
        # initialize states
        dt = 0.01
        time = 0.0
        x = [0, 0, 0]
        w = self.calculate_circle_angular_velocity(0.5, 1.0)
        u = [1.0, w]
        self.landmarks = self.generate_landmarks()

        # simulate two wheel robot
        for i in range(300):
            # update state
            x = two_wheel_2d_model(x, u, dt)

            # convert both euler angles and translation from NWU to EDN
            rpy = nwu2edn([0.0, 0.0, x[2]])
            t = nwu2edn([x[0], x[1], 0.0])

            # check landmark
            observed = self.camera.check_landmarks(dt, self.landmarks, rpy, t)
            if observed is not None:
                self.observed_landmarks.append(observed)
                self.robot_state.append(x)
                self.time.append(time)

            # update
            time += dt

    def generate_test_data(self, save_dir):
        """ Generate test data """
        # mkdir calibration directory
        os.mkdir(save_dir)

        # simulate test data
        self.simulate_test_data()

        # output landmarks and robot state
        self.output_landmarks(save_dir)
        self.output_robot_state(save_dir)
        self.output_observed(save_dir)
