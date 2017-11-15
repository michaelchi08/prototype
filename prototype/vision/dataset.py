import os
from math import pi

import numpy as np

from prototype.utils.utils import rotz
from prototype.utils.utils import nwu2edn
from prototype.utils.data import mat2csv
from prototype.models.two_wheel import two_wheel_2d_model
from prototype.vision.common import camera_intrinsics
from prototype.vision.common import rand3dfeatures
from prototype.vision.camera_model import PinholeCameraModel
from prototype.vision.features import FeatureTrack


DEBUG = False


def debug(s):
    if DEBUG:
        print(s)


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

    counter_frame_id : int
        Counter Frame ID
    counter_track_id : int
        Counter Track ID

    tracks_tracking : :obj`list` of :obj`int`
        List of feature track id
    tracks_lost : :obj`list` of :obj`int`
        List of lost feature track id
    tracks_buffer : :obj`dict` of :obj`FeatureTrack`
        Tracks buffer
    max_buffer_size : int
        Max buffer size (Default: 5000)

    img_ref : np.array
        Reference image
    fea_ref :
        Reference feature
    unmatched : :obj`list` of `Feature`
        List of features

    """

    def __init__(self):
        K = camera_intrinsics(554.25, 554.25, 320.0, 320.0)
        self.cam_model = PinholeCameraModel(640, 640, K, hz=10)
        self.nb_features = 100
        self.feature_bounds = {
            "x": {"min": -10.0, "max": 10.0},
            "y": {"min": -10.0, "max": 10.0},
            "z": {"min": -10.0, "max": 10.0}
        }

        self.features = []
        self.time_history = []
        self.robot_states = []
        self.observed_features = []

        self.pos_true = np.zeros((3, 1))
        self.att_true = np.zeros((3, 1))

        # Counters
        self.counter_frame_id = 0
        self.counter_track_id = 0

        # Feature tracks
        self.landmarks_tracking = []
        self.landmarks_buffer = {}
        self.tracks_tracking = []
        self.tracks_lost = []
        self.tracks_buffer = {}

        # Initialize states
        self.dt = 0.01
        self.time = 0.0

        # Initialize features
        self.features = self.generate_features()

    def get_state(self):
        """Returns robot state

        Returns
        -------
        (pos, v_B, rpy, w_BG) : Tuple of size 4 each is a np.array 3x1

            p : Position in world frame (m)
            v_B : Velocity in body frame (ms^-1)
            rpy : Roll, pitch and yaw in world frame (rads)
            w_BG : Angular velocity in body frame (rads^-1)

        """
        pos = np.block([[self.x[0, 0]], [self.x[1, 0]], [0.0]])
        v_B = np.array([[self.v_B], [0.0], [0.0]])
        rpy = np.array([[0.0], [0.0], [self.x[2]]])
        w_BG = np.array([[0.0], [0.0], [self.w_BG]])

        return pos, v_B, rpy, w_BG

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
        for i in range(len(self.time_history)):
            t = self.time_history[i]
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
        for i in range(len(self.time_history)):
            # Setup output file
            output_path = save_dir + "/observed_" + str(i) + ".dat"
            index_file.write(output_path + '\n')
            obs_file = open(output_path, "w")

            # Data
            t = self.time_history[i]
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

    def detect(self, time, x, dt):
        """Update tracker with current image

        Parameters
        ----------
        time : float
            Time
        x : np.array
            Robot state
        dt : float
            Time difference

        """
        # Convert both euler angles and translation from NWU to EDN
        # Note: We assume here that we are using a two wheel robot model
        # where x = [pos_x, pos_y, theta] in world frame
        rpy = nwu2edn([0.0, 0.0, x[2]])  # Motion model only modelled yaw
        t = nwu2edn([x[0], x[1], 0.0])   # Translation

        # Obtain list of features observed at this time step
        fea_cur = self.cam_model.check_features(dt, self.features, rpy, t)
        if fea_cur is None:
            return

        # Remove lost feature tracks
        landmark_ids_cur = [landmark_id for _, landmark_id in fea_cur]
        for landmark_id in self.landmarks_tracking:
            if landmark_id not in landmark_ids_cur:
                debug("Landmark id: %d is not seen anymore!" % landmark_id)

                track_id = self.landmarks_buffer[landmark_id]
                track = self.tracks_buffer[track_id]
                self.tracks_lost.append(track_id)
                debug("Remove [track_id: %d, landmark_id: %d]" % (track_id,
                                                                  landmark_id))

                self.landmarks_tracking.remove(landmark_id)
                del self.landmarks_buffer[landmark_id]
                del self.tracks_buffer[track_id]
                self.tracks_tracking.remove(track_id)

        # Add or update feature tracks
        for feature in fea_cur:
            kp, landmark_id = feature

            if landmark_id not in self.landmarks_tracking:
                # Add track
                frame_id = self.counter_frame_id
                track_id = self.counter_track_id
                track = FeatureTrack(frame_id, track_id, kp)
                debug("Add [track_id: %d, landmark_id: %d]" % (track_id,
                                                               landmark_id))

                self.landmarks_tracking.append(landmark_id)
                self.landmarks_buffer[landmark_id] = track_id
                self.tracks_tracking.append(track_id)
                self.tracks_buffer[track_id] = track
                self.counter_track_id += 1

            elif landmark_id in self.landmarks_tracking:
                # Update track
                track_id = self.landmarks_buffer[landmark_id]
                track = self.tracks_buffer[track_id]
                track.update(self.counter_frame_id, kp)
                debug("Update [track_id: %d, landmark_id: %d]" % (track_id,
                                                                  landmark_id))

        debug("tracks_tracking: {}".format(self.tracks_tracking))
        debug("landmarks_tracking: {}\n".format(self.landmarks_tracking))
        self.counter_frame_id += 1

        # Keep track of features, robot state and time
        self.observed_features.append(fea_cur)
        self.robot_states.append(x)
        self.time_history.append(time)


    def remove_lost_tracks(self):
        """Remove lost tracks"""
        lost_tracks = []

        # Remove tracks from self.tracks_buffer
        for track_id in self.tracks_lost:
            track = self.tracks_buffer[track_id]
            lost_tracks.append(track)
            del self.tracks_buffer[track_id]

        # Reset tracks lost array
        self.tracks_lost = []

        return lost_tracks

    def step(self):
        """Step

        Returns
        -------
        (a_B, w_BG) : (np.array 3x1, np.array 3x1)
            Accelerometer and Gyroscope measurement in body frame (mimicks IMU
            measurements)

        """
        # Update state


        # Check feature
        self.detect(self.time, self.x, self.dt)

        # Update
        pos = np.block([[self.x[0, 0]], [self.x[1, 0]], [0.0]])
        att = np.block([[0.0], [0.0], [self.x[2, 0]]])
        self.pos_true = np.hstack((self.pos_true, pos))
        self.att_true = np.hstack((self.att_true, att))

        self.time += self.dt

        return (a_B, w_BG)

    def simulate_test_data(self):
        """Simulate test data"""
        for i in range(300):
            self.step()

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
