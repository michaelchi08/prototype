import os
import copy

import numpy as np
from numpy import dot
import matplotlib.pylab as plt

from prototype.utils.euler import euler2rot as R
from prototype.utils.quaternion.jpl import quat2rot as C
from prototype.utils.transform import T_global_camera
from prototype.utils.transform import T_camera_global
from prototype.vision.common import camera_intrinsics
from prototype.vision.common import rand3dfeatures
from prototype.vision.camera.camera_model import PinholeCameraModel
from prototype.vision.feature2d.keypoint import KeyPoint
from prototype.vision.feature2d.feature_track import FeatureTrack


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

        # p_G_f[0] += 0.1
        # p_G_f[1] -= 0.1
        # p_G_f[2] += 0.1

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
            z_hat = np.array([h[0] / h[2], h[1] / h[2]])

            # Calculate reprojection error
            # -- Convert measurment to normalized pixel coordinates
            z = cam_model.pixel2image(track.track[i].pt).reshape((2, 1))
            # -- Reprojection error
            reprojection_error = z - z_hat
            r[2 * i:(2 * (i + 1))] = reprojection_error

        # Plot
        # plot=True
        if plot:
            estimates = np.array(estimates)
            self.plot(track, track_cam_states, estimates)

        # Convert estimated inverse depth params back to feature position in
        # global frame.  See (Eq.38, Mourikis2007 (A Multi-State Constraint
        # Kalman Filter for Vision-aided Inertial Navigation)
        # z = 1 / rho
        # X = np.array([[alpha], [beta], [1.0]])
        # p_G_f = z * dot(C_C0G.T, X) + p_G_C0
        # p_G_f[0] += np.random.normal(0.0, 0.01)
        # p_G_f[1] += np.random.normal(0.0, 0.01)
        # p_G_f[2] += np.random.normal(0.0, 0.01)

        return p_G_f


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
        self.nb_features = kwargs.get("nb_features", 1000)
        self.feature_bounds = {"x": {"min": -10.0, "max": 10.0},
                               "y": {"min": -10.0, "max": 10.0},
                               "z": {"min": 5.0, "max": 20.0}}
        self.features = rand3dfeatures(self.nb_features, self.feature_bounds)

        # Simulation settings
        self.dt = kwargs.get("dt", 0.01)
        self.t = 0.0

        # Linear model
        self.a_B = np.array([[0.01], [0.0], [0.0]])
        self.w_B = np.array([[0.0], [0.0], [0.0]])

        self.pos = np.zeros((3, 1))
        self.vel = np.zeros((3, 1))
        self.acc = self.a_B

        self.att = np.zeros((3, 1))
        self.avel = np.zeros((3, 1))

        # State history
        self.time_true = np.array([0.0])
        self.pos_true = np.zeros((3, 1))
        self.vel_true = np.zeros((3, 1))
        self.acc_true = self.a_B
        self.att_true = np.zeros((3, 1))

        # Counters
        self.counter_frame_id = 0
        self.counter_track_id = 0

        # Feature tracks
        self.features_tracking = []
        self.features_buffer = {}
        self.tracks_tracking = []
        self.tracks_lost = []
        self.tracks_buffer = {}

        self.detect(self.pos, self.att)

    def debug(self, string):
        if self.debug_mode:
            print(string)

    def add_feature_track(self, feature_id, kp, pos, rpy):
        """Add feature track

        Parameters
        ----------
        feature_id : int
            Feature id
        kp : KeyPoint
            KeyPoint

        """
        frame_id = self.counter_frame_id
        track_id = self.counter_track_id

        ground_truth = self.get_feature_position(feature_id)
        ground_truth = T_global_camera * ground_truth
        track = FeatureTrack(track_id,
                             frame_id,
                             kp,
                             ground_truth=ground_truth,
                             pos=[pos],
                             rpy=[rpy])

        self.features_tracking.append(feature_id)
        self.features_buffer[feature_id] = track_id
        self.tracks_tracking.append(track_id)
        self.tracks_buffer[track_id] = track
        self.counter_track_id += 1

        self.debug("+ [track_id: %d, feature_id: %d]" % (track_id, feature_id))

    def remove_feature_track(self, feature_id):
        """Remove feature track

        Parameters
        ----------
        feature_id : int
            Feature id

        """
        track_id = self.features_buffer.pop(feature_id)
        track = self.tracks_buffer.pop(track_id)

        self.features_tracking.remove(feature_id)
        self.tracks_tracking.remove(track_id)
        self.tracks_lost.append(track)

        self.debug("- [track_id: %d, feature_id: %d]" % (track_id, feature_id))

    def update_feature_track(self, feature_id, kp, pos, rpy):
        """Update feature track

        Parameters
        ----------
        feature_id : int
            Feature id
        kp : KeyPoint
            KeyPoint

        """
        track_id = self.features_buffer[feature_id]
        track = self.tracks_buffer[track_id]

        if track.tracked_length() > 20:
            self.remove_feature_track(feature_id)
        else:
            track.update(self.counter_frame_id, kp, pos, rpy)
            self.debug("Update [track_id: %d, feature_id: %d]" %
                       (track_id, feature_id))

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
                self.remove_feature_track(feature_id)

        # Add or update feature tracks
        for feature in fea_cur:
            kp, feature_id = feature
            kp = KeyPoint(kp, 0)

            if feature_id not in self.features_tracking:
                self.add_feature_track(feature_id, kp, pos, rpy)
            else:
                self.update_feature_track(feature_id, kp, pos, rpy)

        self.debug("tracks_tracking: {}".format(self.tracks_tracking))
        self.debug("features_tracking: {}\n".format(self.features_tracking))
        self.counter_frame_id += 1

    def remove_lost_tracks(self):
        """Remove lost tracks"""
        if len(self.tracks_lost) == 0:
            return []

        # Make a copy of current lost tracks
        lost_tracks = []
        for track in self.tracks_lost:
            lost_tracks.append(track)

        # Reset tracks lost array
        self.tracks_lost = []

        # Filter and return lost tracks
        if len(lost_tracks) > 20:
            return lost_tracks[:20]
        else:
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
        self.pos = self.pos + self.vel * self.dt
        self.vel = self.vel + self.acc * self.dt
        self.a_B = self.a_B + np.random.normal(0.0, 0.001, 3).reshape((3, 1))
        self.w_B = self.w_B + np.random.normal(0.0, 0.001, 3).reshape((3, 1))
        self.acc = dot(R(self.att, 321), self.a_B)
        self.att = self.att + dot(R(self.att, 321), self.w_B) * self.dt

        # Check feature
        self.detect(self.pos, self.att)

        # Update
        self.t += self.dt

        # Keep track
        self.time_true = np.hstack((self.time_true, self.t))
        self.pos_true = np.hstack((self.pos_true, self.pos))
        self.vel_true = np.hstack((self.vel_true, self.vel))
        self.acc_true = np.hstack((self.acc_true, self.a_B))
        self.att_true = np.hstack((self.att_true, self.att))

        imu_accel = self.a_B + np.random.normal(0.0, 0.05)
        imu_gyro = self.w_B + np.random.normal(0.0, 0.05)
        # imu_accel = self.a_B
        # imu_gyro = self.w_B

        return (imu_accel, imu_gyro)

    def simulate_test_data(self):
        """Simulate test data"""
        for i in range(300):
            self.step()
