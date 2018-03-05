from math import pi
from math import sqrt
from math import radians

import numpy as np
from numpy import dot
import matplotlib.pyplot as plt

from prototype.utils.euler import euler2rot
from prototype.models.gimbal import GimbalModel

from prototype.vision.common import focal_length
from prototype.vision.common import camera_intrinsics
from prototype.vision.camera.camera_model import PinholeCameraModel

from prototype.calibration.chessboard import Chessboard
from prototype.calibration.preprocess import PreprocessData
from prototype.calibration.camera_intrinsics import CameraIntrinsics

from prototype.viz.common import axis_equal_3dplot
from prototype.viz.plot_gimbal import PlotGimbal
from prototype.viz.plot_chessboard import PlotChessboard


class GimbalDataGenerator:
    def __init__(self, intrinsics_file):
        self.intrinsics = CameraIntrinsics(intrinsics_file)

        # Chessboard
        self.chessboard = Chessboard(t_G=np.array([0.3, 0.1, 0.1]),
                                     nb_rows=11,
                                     nb_cols=11,
                                     square_size=0.02)
        self.plot_chessboard = PlotChessboard(chessboard=self.chessboard)

        # Gimbal
        self.gimbal = GimbalModel()
        self.gimbal.set_attitude([0.0, 0.0])
        self.plot_gimbal = PlotGimbal(gimbal=self.gimbal)

        # Cameras
        self.static_camera = self.setup_static_camera()
        self.gimbal_camera = self.setup_gimbal_camera()

    def setup_static_camera(self):
        image_width = 640
        image_height = 480
        fov = 120
        fx, fy = focal_length(image_width, image_height, fov)
        cx, cy = (image_width / 2.0, image_height / 2.0)
        K = camera_intrinsics(fx, fy, cx, cy)
        cam_model = PinholeCameraModel(image_width, image_height, K)

        return cam_model

    def setup_gimbal_camera(self):
        image_width = 640
        image_height = 480
        fov = 120
        fx, fy = focal_length(image_width, image_height, fov)
        cx, cy = (image_width / 2.0, image_height / 2.0)
        K = camera_intrinsics(fx, fy, cx, cy)
        cam_model = PinholeCameraModel(image_width, image_height, K)

        return cam_model

    def calc_static_camera_view(self):
        # Transforming chessboard grid points in global to camera frame
        R = np.eye(3)
        t = np.zeros(3)
        R_CG = euler2rot([-pi / 2.0, 0.0, -pi / 2.0], 123)
        X = dot(R_CG, self.chessboard.grid_points3d.T)
        x = self.static_camera.project(X, R, t).T[:, 0:2]

        return x

    def calc_gimbal_camera_view(self):
        # Create transform from global to static camera frame
        t_g_sg = np.array([0.0, 0.0, 0.0])
        R_sg = euler2rot([-pi / 2.0, 0.0, -pi / 2.0], 321)
        T_gs = np.array([[R_sg[0, 0], R_sg[0, 1], R_sg[0, 2], t_g_sg[0]],
                         [R_sg[1, 0], R_sg[1, 1], R_sg[1, 2], t_g_sg[1]],
                         [R_sg[2, 0], R_sg[2, 1], R_sg[2, 2], t_g_sg[2]],
                         [0.0, 0.0, 0.0, 1.0]])

        # Calculate transform from global to dynamic camera frame
        links = self.gimbal.calc_transforms()
        T_sd = links[-1]
        T_gd = dot(T_gs, T_sd)

        # Project chessboard grid points in global to dynamic camera frame
        # -- Convert 3D points to homogeneous coordinates
        X = self.chessboard.grid_points3d.T
        X = np.block([[X], [np.ones(X.shape[1])]])
        # -- Project to dynamica camera image frame
        X = dot(np.linalg.inv(T_gd), X)[0:3, :]
        x = dot(self.gimbal_camera.K, X)
        # -- Normalize points
        x[0, :] = x[0, :] / x[2, :]
        x[1, :] = x[1, :] / x[2, :]
        x = x[0:2, :].T

        return x, X.T

    def plot_static_camera_view(self, ax):
        x = self.calc_static_camera_view()
        ax.scatter(x[:, 0], x[:, 1], marker="o", color="red")

    def plot_gimbal_camera_view(self, ax):
        x, X = self.calc_gimbal_camera_view()
        ax.scatter(x[:, 0], x[:, 1], marker="o", color="red")

    def plot_camera_views(self):
        # Plot static camera view
        ax = plt.subplot(211)
        ax.axis('square')
        self.plot_static_camera_view(ax)
        ax.set_title("Static Camera View", y=1.08)
        ax.set_xlim((0, self.static_camera.image_width))
        ax.set_ylim((0, self.static_camera.image_height))
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        # Plot gimbal camera view
        ax = plt.subplot(212)
        ax.axis('square')
        self.plot_gimbal_camera_view(ax)
        ax.set_title("Gimbal Camera View", y=1.08)
        ax.set_xlim((0, self.gimbal_camera.image_width))
        ax.set_ylim((0, self.gimbal_camera.image_height))
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        # Overall plot settings
        plt.tight_layout()

    def plot(self):
        # Plot camera views
        self.plot_camera_views()

        # Plot gimbal and chessboard
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        self.plot_gimbal.plot(ax)
        self.plot_chessboard.plot(ax)
        axis_equal_3dplot(ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

    def calc_roll_pitch_combo(self, nb_images):
        nb_combo = int(sqrt(nb_images))

        roll_lim = [radians(-10), radians(10)]
        pitch_lim = [radians(-10), radians(10)]

        roll_vals = np.linspace(roll_lim[0], roll_lim[1], num=nb_combo)
        pitch_vals = np.linspace(pitch_lim[0], pitch_lim[1], num=nb_combo)

        return roll_vals, pitch_vals

    def generate(self):
        # Setup
        nb_images = 16
        R_CG = euler2rot([-pi / 2.0, 0.0, -pi / 2.0], 123)

        # Generate static camera data
        self.intrinsics.K_new = self.intrinsics.K()
        static_cam_data = PreprocessData("IMAGES",
                                         images_dir=None,
                                         intrinsics=self.intrinsics,
                                         chessboard=self.chessboard)
        x = self.calc_static_camera_view()
        X = dot(R_CG, self.chessboard.grid_points3d.T).T
        for i in range(nb_images):
            static_cam_data.corners2d_ud.append(x)
            static_cam_data.corners3d_ud.append(X)
        static_cam_data.corners2d_ud = np.array(static_cam_data.corners2d_ud)
        static_cam_data.corners3d_ud = np.array(static_cam_data.corners3d_ud)

        # Generate gimbal data
        roll_vals, pitch_vals = self.calc_roll_pitch_combo(nb_images)
        gimbal_cam_data = PreprocessData("IMAGES",
                                         images_dir=None,
                                         intrinsics=self.intrinsics,
                                         chessboard=self.chessboard)
        joint_data = []
        for roll in roll_vals:
            for pitch in pitch_vals:
                self.gimbal.set_attitude([roll, pitch])
                x, X = self.calc_gimbal_camera_view()
                gimbal_cam_data.corners2d_ud.append(x)
                gimbal_cam_data.corners3d_ud.append(X)
                joint_data.append([roll, pitch])

        gimbal_cam_data.corners2d_ud = np.array(gimbal_cam_data.corners2d_ud)
        gimbal_cam_data.corners3d_ud = np.array(gimbal_cam_data.corners3d_ud)
        joint_data = np.array(joint_data)

        # Setup measurement sets
        Z = []
        for i in range(nb_images):
            # Corners 3d observed in both the static and dynamic cam
            P_s = static_cam_data.corners3d_ud[i]
            P_d = gimbal_cam_data.corners3d_ud[i]
            # Corners 2d observed in both the static and dynamic cam
            Q_s = static_cam_data.corners2d_ud[i]
            Q_d = gimbal_cam_data.corners2d_ud[i]
            Z_i = [P_s, P_d, Q_s, Q_d]
            Z.append(Z_i)
        K_s = static_cam_data.intrinsics.K_new
        K_d = gimbal_cam_data.intrinsics.K_new

        # Distortion - assume no distortion
        D_s = np.zeros((4,))
        D_d = np.zeros((4,))

        return Z, K_s, K_d, D_s, D_d, joint_data
