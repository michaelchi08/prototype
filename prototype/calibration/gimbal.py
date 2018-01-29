from os.path import join

from math import pi
from math import sqrt
from math import radians

import cv2
import numpy as np
from numpy import dot
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from prototype.utils.filesystem import walkdir
from prototype.utils.euler import euler2rot
from prototype.models.gimbal import GimbalModel

from prototype.vision.common import focal_length
from prototype.vision.common import camera_intrinsics
from prototype.vision.camera.camera_model import PinholeCameraModel

from prototype.calibration.chessboard import Chessboard
from prototype.calibration.camera_intrinsics import CameraIntrinsics

from prototype.viz.common import axis_equal_3dplot
from prototype.viz.plot_gimbal import PlotGimbal
from prototype.viz.plot_chessboard import PlotChessboard


class ECData:
    """ Extrinsics calibration data

    Parameters
    ----------
    cam0_corners2d : np.array
        Camera 0 image corners
    cam0_corners3d : np.array
        Camera 0 image point location
    imu_data : np.array
        IMU data
    cam0_intrinsics : CameraIntrinsics
        Camera 0 Intrinsics

    """
    def __init__(self, image_dir, intrinsics, chessboard):
        self.image_dir = image_dir
        self.images = []
        self.images_ud = []

        self.intrinsics = intrinsics
        self.chessboard = chessboard
        self.corners2d = []
        self.corners3d = []
        self.corners2d_ud = []
        self.corners3d_ud = []

    def ideal2pixel(self, points, K):
        """ Ideal points to pixel coordinates

        Parameters
        ----------
        cam_id : int
            Camera ID
        points : np.array
            Points in ideal coordinates

        Returns
        -------
        pixels : np.array
            Points in pixel coordinates

        """
        # Get camera intrinsics
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # Convert ideal points to pixel coordinates
        pixels = []
        nb_points = len(points)
        for p in points.reshape((nb_points, 2)):
            px = (p[0] * fx) + cx
            py = (p[1] * fy) + cy
            pixels.append([px, py])

        return np.array(pixels)

    def get_viz(self, i):
        """ Return a visualization of the original and undistorted image with
        detected chessboard corners and a 3D coordinate axis drawn on the
        images.  The original and undistorted image with the visualizations
        will be stacked horizontally.

        Parameters
        ----------
        i : int
            i-th Image frame

        Returns
        -------
        image_viz : np.array
            Image visualization

        """
        # Visualize original image
        image = self.images[i]
        corners2d = self.corners2d[i]
        K = self.intrinsics.K()
        image = self.chessboard.draw_viz(image, corners2d, K)

        # Visualize undistorted image
        image_ud = self.images_ud[i]
        corners2d_ud = self.corners2d_ud[i]
        K_new = self.intrinsics.K_new
        image_ud = self.chessboard.draw_viz(image_ud, corners2d_ud, K_new)

        # Create visualization
        image_viz = np.hstack((image, image_ud))
        return image_viz

    def load(self):
        """ Load extrinsics calibration data """
        image_files = walkdir(self.image_dir)
        nb_images = len(image_files)

        # Loop through calibration images
        for i in range(nb_images):
            # Load images and find chessboard corners
            image = cv2.imread(image_files[i])
            corners = self.chessboard.find_corners(image)
            self.images.append(image)

            # Calculate camera to chessboard transform
            K = self.intrinsics.K()
            P_c = self.chessboard.calc_corner_positions(corners, K)
            nb_corners = corners.shape[0]
            self.corners2d.append(corners.reshape((nb_corners, 2)))
            self.corners3d.append(P_c)

            # Undistort corners in camera 0
            corners_ud = self.intrinsics.undistort_points(corners)
            image_ud, K_new = self.intrinsics.undistort_image(image)
            pixels_ud = self.ideal2pixel(corners_ud, K_new)
            self.images_ud.append(image_ud)

            # Calculate camera to chessboard transform
            K_new = self.intrinsics.K_new
            P_c = self.chessboard.calc_corner_positions(pixels_ud, K_new)
            self.corners2d_ud.append(pixels_ud)
            self.corners3d_ud.append(P_c)

        self.corners2d = np.array(self.corners2d)
        self.corners3d = np.array(self.corners3d)
        self.corners2d_ud = np.array(self.corners2d_ud)
        self.corners3d_ud = np.array(self.corners3d_ud)


class GECDataLoader:
    """ Gimbal extrinsics calibration data loader

    Attributes
    ----------
    data_path : str
        Data path
    cam0_dir : str
        Camera 0 image dir
    cam1_dir : str
        Camera 1 image dir
    imu_filename : str
        IMU data path
    chessboard : Chessboard
        Chessboard
    imu_data : np.array
        IMU data

    """
    def __init__(self, **kwargs):
        self.data_path = kwargs.get("data_path")
        self.image_dirs = kwargs["image_dirs"]
        self.intrinsic_files = kwargs["intrinsic_files"]
        self.imu_file = kwargs["imu_file"]
        self.chessboard = Chessboard(**kwargs)

        self.inspect_data = kwargs.get("inspect_data", False)

    def load_imu_data(self):
        """ Load IMU data

        Parameters
        ----------
        imu_fpath : str
            IMU data file path

        Returns
        -------
        imu_data : np.array
            IMU data

        """
        imu_file = open(join(self.data_path, self.imu_file), "r")
        imu_data = np.loadtxt(imu_file, delimiter=",")
        imu_file.close()
        return imu_data

    def draw_corners(self, image, corners, color=(0, 255, 0)):
        """ Draw corners

        Parameters
        ----------
        image : np.array
            Image
        corners : np.array
            Corners

        """
        image = np.copy(image)
        for i in range(len(corners)):
            corner = tuple(corners[i][0].astype(int).tolist())
            image = cv2.circle(image, corner, 2, color, -1)
        return image

    def check_nb_images(self, data):
        """ Check number of images in data """
        nb_cameras = len(self.image_dirs)

        nb_images = len(data[0].images)
        for i in range(1, nb_cameras):
            if len(data[i].images) != nb_images:
                err = "Number of images mismatch! [{0}] - [{1}]".format(
                    self.image_dirs[0],
                    self.image_dirs[i]
                )
                raise RuntimeError(err)

        return True

    def load(self):
        """ Load calibration data """
        # Load imu data
        imu_data = self.load_imu_data()

        # Load camera data
        nb_cameras = len(self.image_dirs)
        ec_data = []
        for i in range(nb_cameras):
            image_dir = join(self.data_path, self.image_dirs[i])
            intrinsics_file = join(self.data_path, self.intrinsic_files[i])
            intrinsics = CameraIntrinsics(intrinsics_file)
            data_entry = ECData(image_dir, intrinsics, self.chessboard)
            data_entry.load()
            ec_data.append(data_entry)

        # Inspect data
        self.check_nb_images(ec_data)
        if self.inspect_data is False:
            return ec_data, imu_data

        nb_images = len(ec_data[0].images)
        for i in range(nb_images):
            viz = ec_data[0].get_viz(i)
            for n in range(1, nb_cameras):
                viz = np.vstack((viz, ec_data[n].get_viz(i)))
            cv2.imshow("Image", viz)
            cv2.waitKey(0)

        return ec_data, imu_data


class GEC:
    """ Gimbal Extrinsics Calibrator

    Attributes
    ----------
    gimbal_model : GimbalModel
        Gimbal model
    data : GECDataLoader
        Calibration data

    """
    def __init__(self, **kwargs):
        self.gimbal_model = GimbalModel()

        if kwargs.get("sim_mode", False):
            self.ec_data = kwargs["ec_data"]
            self.imu_data = kwargs["imu_data"]
        else:
            self.loader = GECDataLoader(**kwargs)
            self.ec_data, self.imu_data = self.loader.load()

    def setup_problem(self):
        """ Setup the calibration optimization problem

        Returns
        -------
        x : np.array
            Vector of optimization parameters to be optimized

        """
        print("Setting up optimization problem ...")

        # Setup vector of parameters to be optimized
        K = len(self.ec_data[0].corners2d_ud)  # Number of measurement set
        L = 2  # Number of links - 2 for a 2-axis gimbal
        x = np.zeros(6 + 6 + 3 * L + K * L)  # Parameters to be optimized

        x[0:6] = self.gimbal_model.tau_s
        x[6:12] = self.gimbal_model.tau_d
        x[12:15] = self.gimbal_model.w1
        x[15:18] = self.gimbal_model.w2
        x[18:18+K*L] = self.imu_data[:, 0:L].ravel()

        # Setup measurement sets
        Z = []
        for i in range(K):
            # Corners 3d observed in both the static and dynamic cam
            P_s = self.ec_data[0].corners3d_ud[i]
            P_d = self.ec_data[1].corners3d_ud[i]
            # Corners 2d observed in both the static and dynamic cam
            Q_s = self.ec_data[0].corners2d_ud[i]
            Q_d = self.ec_data[1].corners2d_ud[i]
            Z_i = [P_s, P_d, Q_s, Q_d]
            Z.append(Z_i)

        # Get camera matrix K
        K_s = self.ec_data[0].intrinsics.K_new
        K_d = self.ec_data[1].intrinsics.K_new

        return x, Z, K_s, K_d

    def reprojection_error(self, x, *args):
        """Reprojection Error

        Parameters
        ----------
        x : np.array
            Parameters to be optimized
        args : tuple of (Z, K_s, K_d)
            Z: list of measurement sets
            K_s: np.array static camera intrinsics matrix K
            K_d: np.array dynamic camera intrinsics matrix K

        Returns
        -------
        residual : np.array
            Reprojection error

        """
        L = 2  # Number of links
        K = (len(x) - 18) / L  # Number of set measurements
        tau_s = x[0:6]
        tau_d = x[6:12]
        w1 = x[12:15]
        w2 = x[15:18]

        Z, K_s, K_d = args

        # Loop through all measurement sets
        residuals = []
        for k in range(int(K)):
            # Get joint angles
            Lambda1 = x[18 + (k * 2)]  # Roll
            Lambda2 = x[18 + (k * 2 + 1)]  # Pitch

            # Get the k-th measurements
            P_s, P_d, Q_s, Q_d = Z[k]

            # Calculate static to dynamic camera transform
            T_sd = self.gimbal_model.T_sd(tau_s,
                                          Lambda1, w1, Lambda2, w2,
                                          tau_d)

            # Calculate reprojection error in the static camera
            nb_P_d_corners = len(P_d)
            err_s = np.zeros(nb_P_d_corners * 2)
            for i in range(nb_P_d_corners):
                # -- Transform 3D world point from dynamic to static camera
                P_d_homo = np.append(P_d[i], 1.0)
                P_s_cal = dot(T_sd, P_d_homo)[0:3]
                # -- Project 3D world point to image plane
                Q_s_cal = dot(K_s, P_s_cal)
                # -- Normalize projected image point
                Q_s_cal[0] = Q_s_cal[0] / Q_s_cal[2]
                Q_s_cal[1] = Q_s_cal[1] / Q_s_cal[2]
                Q_s_cal = Q_s_cal[:2]
                # -- Calculate reprojection error
                err_s[(i * 2):(i * 2 + 2)] = Q_s[i] - Q_s_cal

            # Calculate reprojection error in the dynamic camera
            nb_P_s_corners = len(P_s)
            err_d = np.zeros(nb_P_s_corners * 2)
            for i in range(nb_P_s_corners):
                # -- Transform 3D world point from dynamic to static camera
                P_s_homo = np.append(P_s[i], 1.0)
                P_d_cal = dot(np.linalg.inv(T_sd), P_s_homo)[0:3]
                # -- Project 3D world point to image plane
                Q_d_cal = dot(K_d, P_d_cal)
                # -- Normalize projected image point
                Q_d_cal[0] = Q_d_cal[0] / Q_d_cal[2]
                Q_d_cal[1] = Q_d_cal[1] / Q_d_cal[2]
                Q_d_cal = Q_d_cal[:2]
                # -- Calculate reprojection error
                err_d[(i * 2):(i * 2 + 2)] = Q_d[i] - Q_d_cal

            # Stack residuals
            residuals.append(np.block([err_s, err_d]))

        return np.array(residuals).reshape((-1))

    def optimize(self):
        """ Optimize Gimbal Extrinsics """
        # Setup
        x, Z, K_s, K_d = self.setup_problem()
        args = [Z, K_s, K_d]

        # Optimize
        print("Optimizing!")
        print("This can take a while...")
        result = least_squares(self.reprojection_error,
                               x,
                               args=args,
                               verbose=1)

        # Parse results
        self.gimbal_model.tau_s = result.x[0:6]
        self.gimbal_model.tau_d = result.x[6:12]
        self.gimbal_model.w1 = result.x[12:15]
        self.gimbal_model.w2 = result.x[15:18]
        self.gimbal_model.Lambda1 = 0
        self.gimbal_model.Lambda2 = 0

        print("Results:")
        print("---------------------------------")
        print("tau_s: ", self.gimbal_model.tau_s)
        print("tau_d: ", self.gimbal_model.tau_d)
        print("w1: ", self.gimbal_model.w1)
        print("w2: ", self.gimbal_model.w2)

        # Plot gimbal
        plot_gimbal = PlotGimbal(gimbal=self.gimbal_model)
        plot_gimbal.plot()
        plt.show()


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
        self.gimbal.set_attitude([0.0, 0.0, 0.0])
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
        nb_images = 100
        R_CG = euler2rot([-pi / 2.0, 0.0, -pi / 2.0], 123)

        # Generate static camera data
        self.intrinsics.K_new = self.intrinsics.K()
        static_cam_data = ECData(None, self.intrinsics, self.chessboard)
        x = self.calc_static_camera_view()
        X = dot(R_CG, self.chessboard.grid_points3d.T).T
        for i in range(nb_images):
            static_cam_data.corners2d_ud.append(x)
            static_cam_data.corners3d_ud.append(X)
        static_cam_data.corners2d_ud = np.array(static_cam_data.corners2d_ud)
        static_cam_data.corners3d_ud = np.array(static_cam_data.corners3d_ud)

        # Generate gimbal data
        roll_vals, pitch_vals = self.calc_roll_pitch_combo(nb_images)
        gimbal_cam_data = ECData(None, self.intrinsics, self.chessboard)
        imu_data = []

        for roll in roll_vals:
            for pitch in pitch_vals:
                self.gimbal.set_attitude([roll, pitch])
                x, X = self.calc_gimbal_camera_view()
                gimbal_cam_data.corners2d_ud.append(x)
                gimbal_cam_data.corners3d_ud.append(X)
                imu_data.append([roll, pitch])

        gimbal_cam_data.corners2d_ud = np.array(gimbal_cam_data.corners2d_ud)
        gimbal_cam_data.corners3d_ud = np.array(gimbal_cam_data.corners3d_ud)
        imu_data = np.array(imu_data)

        return [static_cam_data, gimbal_cam_data], imu_data
