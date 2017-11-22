import os
import copy

import numpy as np
from numpy import dot
import matplotlib.pylab as plt

# from prototype.utils.data import mat2csv
from prototype.utils.quaternion.jpl import quat2rot as C
from prototype.utils.transform import T_global_camera
from prototype.utils.transform import T_camera_global
from prototype.models.husky import HuskyModel
from prototype.control.utils import circle_trajectory
from prototype.vision.common import camera_intrinsics
from prototype.vision.common import rand3dfeatures
from prototype.vision.camera_model import PinholeCameraModel
from prototype.vision.features import Keypoint
from prototype.vision.features import FeatureTrack


class DatasetFeatureEstimator:
    def plot(self, track, track_cam_states, estimates):
        plt.figure()

        # Feature
        feature = T_global_camera * track.ground_truth
        plt.plot(feature[0], feature[1],
                 marker="o", color="red", label="feature")

        # Camera states
        for cam_state in track_cam_states:
            pos = T_global_camera * cam_state.p_G
            plt.plot(pos[0], pos[1],
                     marker="o", color="blue", label="camera")

        # Estimates
        for i in range(len(estimates)):
            cam_state = track_cam_states[i]
            cam_pos = T_global_camera * cam_state.p_G
            estimate = (T_global_camera * estimates[i]) + cam_pos
            plt.plot(estimate[0], estimate[1],
                     marker="o", color="green")

        plt.legend(loc=0)
        plt.show()

    def estimate(self, cam_model, track, track_cam_states, plot=False):
        # Get ground truth
        p_G_f = track.ground_truth

        # Convert ground truth expressed in global frame
        # to be expressed in camera 0
        C_C0G = C(track_cam_states[0].q_CG)
        p_G_C0 = track_cam_states[0].p_G
        p_C0_f = dot(C_C0G, (p_G_f - p_G_C0))

        # Create inverse depth params (these are to be optimized)
        alpha = p_C0_f[0, 0] / p_C0_f[2, 0]
        beta = p_C0_f[1, 0] / p_C0_f[2, 0]
        rho = 1.0 / p_C0_f[2, 0]

        # Setup residual calculation
        N = len(track_cam_states)
        r = np.zeros((2 * N, 1))

        # Calculate residuals
        estimates = []
        for i in range(N):
            # Get camera current rotation and translation
            C_CiG = C(track_cam_states[i].q_CG)
            p_G_Ci = track_cam_states[i].p_G

            # Set camera 0 as origin, work out rotation and translation
            # of camera i relative to to camera 0
            C_CiC0 = dot(C_CiG, C_C0G.T)
            t_Ci_CiC0 = dot(C_CiG, (p_G_C0 - p_G_Ci))

            # Project estimated feature location to image plane
            h = dot(C_CiC0, np.array([[alpha], [beta], [1]])) + rho * t_Ci_CiC0
            estimates.append(h)
            # -- Convert feature location to normalized pixel coordinates
            x = np.array([h[0] / h[2], h[1] / h[2]])

            # Calculate reprojection error
            # -- Convert measurment to normalized pixel coordinates
            z = cam_model.pixel2image(track.track[i].pt).reshape((2, 1))
            # -- Reprojection error
            reprojection_error = z - x
            r[2 * i:(2 * (i + 1))] = reprojection_error

        # Plot
        if plot:
            estimates = np.array(estimates)
            self.plot(track, track_cam_states, estimates)

        # Convert estimated inverse depth params back to feature position in
        # global frame.  See (Eq.38, Mourikis2007 (A Multi-State Constraint
        # Kalman Filter for Vision-aided Inertial Navigation)
        z = 1 / rho
        X = np.array([[alpha], [beta], [1.0]])
        p_G_f = z * dot(C_C0G.T, X) + p_G_C0

        return (p_G_f, 0, r)


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

    def __init__(self, **kwargs):
        # Debug mode?
        self.debug_mode = kwargs.get("debug_mode", False)

        # Camera
        K = camera_intrinsics(554.25, 554.25, 320.0, 320.0)
        self.cam_model = PinholeCameraModel(640, 640, K)

        # Features
        self.nb_features = kwargs.get("nb_features", 10000)
        self.feature_bounds = {
            "x": {"min": -10.0, "max": 10.0},
            "y": {"min": -10.0, "max": 10.0},
            "z": {"min": 5.0, "max": 10.0}
        }
        self.features = rand3dfeatures(self.nb_features, self.feature_bounds)

        # Simulation settings
        self.dt = kwargs.get("dt", 0.01)
        self.t = 0.0

        # Calculate desired inputs for a circle trajectory
        circle_r = 10.0
        circle_vel = 1.0
        circle_w = circle_trajectory(circle_r, circle_vel)
        self.v_B = np.array([[circle_vel], [0.0], [0.0]])
        self.w_B = np.array([[0.0], [0.0], [circle_w]])

        # Motion model and history
        self.model = HuskyModel(vel=self.v_B)
        self.time_true = np.array([0.0])
        self.pos_true = np.zeros((3, 1))
        self.vel_true = self.v_B
        self.acc_true = (self.v_B + self.v_B * self.dt) - (self.v_B)
        self.rpy_true = np.zeros((3, 1))

        # Counters
        self.counter_frame_id = 0
        self.counter_track_id = 0

        # Feature tracks
        self.features_tracking = []
        self.features_buffer = {}
        self.tracks_tracking = []
        self.tracks_lost = []
        self.tracks_buffer = {}

    def debug(self, string):
        if self.debug_mode:
            print(string)

    # def output_robot_state(self, save_dir):
    #     """Output robot state
    #
    #     Parameters
    #     ----------
    #     save_dir : str
    #         Path to save output
    #
    #     """
    #     # Setup state file
    #     header = ["time_step", "x", "y", "theta"]
    #     state_file = open(os.path.join(save_dir, "state.dat"), "w")
    #     state_file.write(",".join(header) + "\n")
    #
    #     # Write state file
    #     for i in range(len(self.time_true)):
    #         t = self.time_true[i]
    #         pos = self.pos_true[:, i].ravel().tolist()
    #
    #         state_file.write(str(t) + ",")
    #         state_file.write(str(pos[0]) + ",")
    #         state_file.write(str(pos[1]) + ",")
    #         state_file.write(str(pos[2]) + "\n")
    #
    #     # Clean up
    #     state_file.close()

    # def output_observed(self, save_dir):
    #     """Output observed features
    #
    #     Parameters
    #     ----------
    #     save_dir : str
    #         Path to save output
    #
    #     """
    #     # Setup
    #     index_file = open(os.path.join(save_dir, "index.dat"), "w")
    #
    #     # Output observed features
    #     for i in range(len(self.time_true)):
    #         # Setup output file
    #         output_path = save_dir + "/observed_" + str(i) + ".dat"
    #         index_file.write(output_path + '\n')
    #         obs_file = open(output_path, "w")
    #
    #         # Data
    #         t = self.time_true[i]
    #         x = self.pos_true[i].ravel().tolist()
    #         observed = self.observed_features[i]
    #
    #         # Output time, robot state, and number of observed features
    #         obs_file.write(str(t) + '\n')
    #         obs_file.write(','.join(map(str, x)) + '\n')
    #         obs_file.write(str(len(observed)) + '\n')
    #
    #         # Output observed features
    #         for obs in self.observed_features[i]:
    #             img_pt, feature_id = obs
    #
    #             # Convert to string
    #             img_pt = ','.join(map(str, img_pt[0:2]))
    #             feature_id = str(feature_id)
    #
    #             # Write to file
    #             obs_file.write(img_pt + '\n')
    #             obs_file.write(feature_id + '\n')
    #
    #         # Close observed file
    #         obs_file.close()
    #
    #     # Close index file
    #     index_file.close()

    # def output_features(self, save_dir):
    #     """Output features
    #
    #     Parameters
    #     ----------
    #     save_dir : str
    #         Path to save output
    #
    #     """
    #     mat2csv(os.path.join(save_dir, "features.dat"), self.features)

    def detect(self, pos, rpy):
        """Update tracker with current image

        Parameters
        ----------
        pos : np.array
            Robot position (x, y, z)

        rpy : np.array
            Robot attitude (roll, pitch, yaw)

        """
        # Convert both euler angles and translation from NWU to EDN
        # Note: We assume here that we are using a robot model
        # where x = [pos_x, pos_y, theta] in world frame
        rpy = T_camera_global * rpy  # Motion model only modelled yaw
        t = T_camera_global * pos  # Translation

        # Obtain list of features observed at this time step
        fea_cur = self.cam_model.observed_features(self.features, rpy, t)
        if fea_cur is None:
            return

        # Remove lost feature tracks
        feature_ids_cur = [feature_id for _, feature_id in list(fea_cur)]
        for feature_id in list(self.features_tracking):
            if feature_id not in feature_ids_cur:
                track_id = self.features_buffer.pop(feature_id)
                track = self.tracks_buffer.pop(track_id)

                self.features_tracking.remove(feature_id)
                self.tracks_tracking.remove(track_id)
                self.tracks_lost.append(track)

                self.debug(
                    "- [track_id: %d, feature_id: %d]" %
                    (track_id, feature_id)
                )

        # Add or update feature tracks
        for feature in fea_cur:
            kp, feature_id = feature
            kp = Keypoint(kp, 0)

            if feature_id not in self.features_tracking:
                # Add track
                frame_id = self.counter_frame_id
                track_id = self.counter_track_id
                ground_truth = T_global_camera * self.get_feature_position(feature_id)
                track = FeatureTrack(track_id,
                                     frame_id,
                                     kp,
                                     ground_truth=ground_truth)
                self.debug(
                    "+ [track_id: %d, feature_id: %d]" %
                    (track_id, feature_id)
                )

                self.features_tracking.append(feature_id)
                self.features_buffer[feature_id] = track_id
                self.tracks_tracking.append(track_id)
                self.tracks_buffer[track_id] = track
                self.counter_track_id += 1

            else:
                # Update track
                track_id = self.features_buffer[feature_id]
                track = self.tracks_buffer[track_id]
                track.update(self.counter_frame_id, kp)
                self.debug(
                    "Update [track_id: %d, feature_id: %d]" %
                    (track_id, feature_id)
                )

        self.debug("tracks_tracking: {}".format(self.tracks_tracking))
        self.debug("features_tracking: {}\n".format(self.features_tracking))
        self.counter_frame_id += 1

    def remove_lost_tracks(self):
        """Remove lost tracks"""
        if len(self.tracks_lost) == 0:
            return []

        # Make a copy of current lost tracks
        lost_tracks = copy.deepcopy(self.tracks_lost)

        # Reset tracks lost array
        self.tracks_lost = []

        return lost_tracks

    def get_feature_position(self, feature_id):
        """Returns feature position"""
        return self.features[:, feature_id].reshape((3, 1))

    def step(self):
        """Step

        Returns
        -------
        (a_B, w_B) : (np.array 3x1, np.array 3x1)
            Accelerometer and Gyroscope measurement in body frame (mimicks IMU
            measurements)

        """
        # Update motion model
        self.model.update(self.v_B, self.w_B, self.dt)
        pos = self.model.p_G
        rpy = self.model.rpy_G

        # Check feature
        self.detect(pos, rpy)

        # Update
        self.t += self.dt

        # Keep track
        self.time_true = np.hstack((self.time_true, self.t))
        self.pos_true = np.hstack((self.pos_true, self.model.p_G))
        self.vel_true = np.hstack((self.vel_true, self.model.v_G))
        self.acc_true = np.hstack((self.acc_true, self.model.a_G))
        self.rpy_true = np.hstack((self.rpy_true, self.model.rpy_G))

        return (self.model.a_B, self.w_B)

    def estimate(self):
        pass

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
        # self.output_features(save_dir)
        # self.output_robot_state(save_dir)
        # self.output_observed(save_dir)
