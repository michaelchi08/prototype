from os.path import join

import cv2
import numpy as np
from numpy import dot
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from prototype.utils.filesystem import walkdir
from prototype.models.gimbal import GimbalModel
from prototype.calibration.chessboard import Chessboard
from prototype.calibration.camera_intrinsics import CameraIntrinsics
from prototype.viz.plot_gimbal import PlotGimbal


class GECData:
    """ Gimbal extrinsics calibration data

    cam0_corners2d : np.array
        Camera 0 image corners
    cam1_corners2d : np.array
        Camera 1 image corners
    cam0_corners3d : np.array
        Camera 0 image point location
    cam1_corners3d : np.array
        Camera 1 image point location
    imu_data : np.array
        IMU data
    cam0_intrinsics : CameraIntrinsics
        Camera 0 Intrinsics
    cam1_intrinsics : CameraIntrinsics
        Camera 1 Intrinsics

    """
    def __init__(self):
        self.object_points = None
        self.cam0_corners2d = []
        self.cam1_corners2d = []
        self.cam0_corners3d = []
        self.cam1_corners3d = []
        self.imu_data = []

        self.cam0_intrinsics = None
        self.cam1_intrinsics = None


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
    cam_intrinsics : np.array
        Camera intrinsics
    imu_filename : str
        IMU data path
    chessboard : Chessboard
        Chessboard

    cam0_corners2d : np.array
        Camera 0 image corners
    cam1_corners2d : np.array
        Camera 1 image corners
    cam0_corners3d : np.array
        Camera 0 image point location
    cam1_corners3d : np.array
        Camera 1 image point location
    imu_data : np.array
        IMU data
    cam0_intrinsics : CameraIntrinsics
        Camera 0 Intrinsics
    cam1_intrinsics : CameraIntrinsics
        Camera 1 Intrinsics

    """
    def __init__(self, **kwargs):
        self.data_path = kwargs.get("data_path")
        self.cam0_dir = kwargs.get("cam0_dir", "cam0")
        self.cam1_dir = kwargs.get("cam1_dir", "cam1")
        self.intrinsics_filename = kwargs.get("intrinsics_filename",
                                              "intrinsics.yaml")
        self.imu_filename = kwargs.get("imu_filename", "imu.dat")
        self.chessboard = Chessboard(**kwargs)

        self.data = GECData()
        self.data.object_points = self.chessboard.object_points

    def load_imu_data(self, imu_fpath):
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
        imu_file = open(imu_fpath, "r")
        imu_data = np.loadtxt(imu_file, delimiter=",")
        imu_file.close()
        return imu_data

    def load_cam_intrinsics(self, intrinsics_fpath):
        """ Load IMU data

        Parameters
        ----------
        intrinsics_fpath : str
            Intrinsics file path

        """
        self.data.cam0_intrinsics = CameraIntrinsics(0, intrinsics_fpath)
        self.data.cam1_intrinsics = CameraIntrinsics(1, intrinsics_fpath)

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

    def ideal2pixel(self, cam_id, points, K=None):
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
        fx, fy, cx, cy = (None, None, None, None)
        if K is not None:
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
        elif cam_id == 0:
            fx, fy, cx, cy = self.data.cam0_intrinsics.intrinsics
        elif cam_id == 1:
            fx, fy, cx, cy = self.data.cam1_intrinsics.intrinsics

        # Convert ideal points to pixel coordinates
        pixels = []
        nb_points = len(points)
        for p in points.reshape((nb_points, 2)):
            px = (p[0] * fx) + cx
            py = (p[1] * fy) + cy
            pixels.append([px, py])

        return np.array([pixels])

    def draw_coord_frame(self, chessboard, img, corners, K, D=np.zeros(4)):
        # Coordinate frame
        axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)
        axis = chessboard.square_size * axis  # Scale chessboard sq size

        # Solve PnP and project coordinate frame to image plane
        T, rvec, tvec = chessboard.solvepnp(corners, K, D)
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, D)

        # Draw coordinate frame
        corner = tuple(corners[0].ravel())
        corner = (int(corner[0]), int(corner[1]))
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 3)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 3)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 3)

        return img

    def visual_inspect_images(self,
                              cam0_img, cam0_img_ud, K0_new,
                              corners0, pixels0_ud,
                              cam1_img, cam1_img_ud, K1_new,
                              corners1, pixels1_ud):
        # Draw corners in camera 0
        cam0_img = self.draw_corners(cam0_img, corners0)
        cam0_img = self.draw_coord_frame(self.chessboard,
                                         cam0_img,
                                         corners0,
                                         self.data.cam0_intrinsics.K())

        cam0_img_ud = self.draw_corners(cam0_img_ud, pixels0_ud)
        cam0_img_ud = self.draw_coord_frame(self.chessboard,
                                            cam0_img_ud,
                                            pixels0_ud,
                                            K0_new)
        cam0_img_pair = np.hstack((cam0_img, cam0_img_ud))

        # Draw corners in camera 1
        cam1_img = self.draw_corners(cam1_img, corners1)
        cam1_img_ud = self.draw_corners(cam1_img_ud, pixels1_ud)
        cam1_img = self.draw_coord_frame(self.chessboard,
                                         cam1_img,
                                         corners1,
                                         self.data.cam1_intrinsics.K())
        cam1_img_ud = self.draw_coord_frame(self.chessboard,
                                            cam1_img_ud,
                                            pixels1_ud,
                                            K1_new)
        cam1_img_pair = np.hstack((cam1_img, cam1_img_ud))

        # Show camera 0 and camera 1 (detected points, undistored points)
        cam0_cam1_img = np.vstack((cam0_img_pair, cam1_img_pair))
        cv2.imshow("Inspect Images", cam0_cam1_img)

    def calc_chessboard_corner_positions(self, T_o_t):
        """ Calculate chessboard corner position

        Having calculated the passive transform from camera to chessboard via
        SolvePnP, this function takes the transform and calculates the 3D
        position of each chessboard corner.

        Returns
        -------
        X : np.array (Nx3)
            N Chessboard corners in 3D as a matrix

        """
        assert self.data.object_points is not None

        nb_obj_pts = self.data.object_points.shape[0]
        obj_pts_homo = np.block([self.data.object_points,
                                 np.ones((nb_obj_pts, 1))])
        obj_pts_homo = obj_pts_homo.T

        X = np.dot(T_o_t, obj_pts_homo)
        X = X.T
        X = X[:, 0:3]

        return X

    def load(self, imshow=False):
        """ Load calibration data """
        # Load camera image file paths
        cam0_path = join(self.data_path, self.cam0_dir)
        cam1_path = join(self.data_path, self.cam1_dir)
        cam0_files = walkdir(cam0_path)
        cam1_files = walkdir(cam1_path)

        # Load imu data
        imu_fpath = join(self.data_path, self.imu_filename)
        imu_data = self.load_imu_data(imu_fpath)

        # Load camera intrinsics
        intrinsics_fpath = join(self.data_path, self.intrinsics_filename)
        self.load_cam_intrinsics(intrinsics_fpath)

        # Pre-check
        assert len(cam0_files) == len(cam1_files)
        assert len(cam0_files) == imu_data.shape[0]

        # Inspect images and prep calibration data
        nb_images = len(cam0_files)

        for i in range(nb_images):
            print("Inspecting image %d" % i)

            # Load images and find chessboard corners
            cam0_img = cv2.imread(cam0_files[i])
            cam1_img = cv2.imread(cam1_files[i])
            corners0 = self.chessboard.find_corners(cam0_img)
            corners1 = self.chessboard.find_corners(cam1_img)
            if corners0 is None or corners1 is None:
                continue  # Skip this for loop iteration

            # Undistort corners in camera 0
            corners0_ud = self.data.cam0_intrinsics.undistort_points(corners0)
            result = self.data.cam0_intrinsics.undistort_image(cam0_img)
            cam0_img_ud, K0_new = result
            pixels0_ud = self.ideal2pixel(0, corners0_ud, K0_new)

            # Undistort corners in camera 1
            corners1_ud = self.data.cam1_intrinsics.undistort_points(corners1)
            result = self.data.cam1_intrinsics.undistort_image(cam1_img)
            cam1_img_ud, K1_new = result
            pixels1_ud = self.ideal2pixel(1, corners1_ud, K1_new)

            # Visually inspect images
            if imshow:
                self.visual_inspect_images(cam0_img, cam0_img_ud, K0_new,
                                           corners0, pixels0_ud,
                                           cam1_img, cam1_img_ud, K1_new,
                                           corners1, pixels1_ud)
                if cv2.waitKey(0) == 113:
                    return None

            # Calculate camera to chessboard transform
            K0_new = self.data.cam0_intrinsics.K_new
            K1_new = self.data.cam1_intrinsics.K_new
            T_c0_cb, _, _ = self.chessboard.solvepnp(pixels0_ud, K0_new)
            T_c1_cb, _, _ = self.chessboard.solvepnp(pixels1_ud, K1_new)
            P_c0 = self.calc_chessboard_corner_positions(T_c0_cb)
            P_c1 = self.calc_chessboard_corner_positions(T_c1_cb)

            # Append to calibration data
            self.data.cam0_corners2d.append(pixels0_ud[0])
            self.data.cam1_corners2d.append(pixels1_ud[0])
            self.data.cam0_corners3d.append(P_c0)
            self.data.cam1_corners3d.append(P_c1)
            self.data.imu_data.append(imu_data[i, :])

        self.data.cam0_corners2d = np.array(self.data.cam0_corners2d)
        self.data.cam1_corners2d = np.array(self.data.cam1_corners2d)
        self.data.imu_data = np.array(self.data.imu_data)
        if imshow:
            cv2.destroyAllWindows()

        return self.data


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
        self.loader = GECDataLoader(**kwargs)
        self.data = self.loader.load(imshow=kwargs.get("inspect_data", False))

    def setup_problem(self):
        """ Setup the calibration optimization problem

        Returns
        -------
        x : np.array
            Vector of optimization parameters to be optimized

        """
        assert len(self.data.cam0_corners2d) == len(self.data.cam1_corners2d)
        assert len(self.data.cam0_corners3d) == len(self.data.cam0_corners3d)
        print("Setting up optimization problem ...")

        # Setup vector of parameters to be optimized
        K = len(self.data.cam0_corners2d)  # Number of measurement set
        L = 2  # Number of links - 2 for a 2-axis gimbal
        x = np.zeros(6 + 6 + 3 * L + K * L)  # Parameters to be optimized

        x[0:6] = self.gimbal_model.tau_s
        x[6:12] = self.gimbal_model.tau_d
        x[12:15] = self.gimbal_model.w1
        x[15:18] = self.gimbal_model.w2
        x[18:18+K*L] = self.data.imu_data[:, 0:L].ravel()

        # Setup measurement sets
        Z = []
        for i in range(K):
            # Corners 3d observed in both the static and dynamic cam
            P_s = self.data.cam0_corners3d[i]
            P_d = self.data.cam1_corners3d[i]
            # Corners 2d observed in both the static and dynamic cam
            Q_s = self.data.cam0_corners2d[i]
            Q_d = self.data.cam1_corners2d[i]
            Z_i = [P_s, P_d, Q_s, Q_d]
            Z.append(Z_i)

        return x, Z

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

        residuals = []
        for i in range(int(K)):
            # Get joint angles
            Lambda1 = x[18 + (i * 2)]  # Roll
            Lambda2 = x[18 + (i * 2 + 1)]  # Pitch

            # Get measurements
            P_s, P_d, Q_s, Q_d = Z[i]

            # Calculate static to dynamic camera transform
            T_sd = self.gimbal_model.T_sd(tau_s,
                                          Lambda1, w1, Lambda2, w2,
                                          tau_d)

            # Calculate reprojection error in the static camera
            nb_P_d_corners = len(P_d)
            err_s = np.zeros(nb_P_d_corners * 2)
            for j in range(nb_P_d_corners):
                # -- Transform 3D world point from dynamic to static camera
                P_d_homo = np.append(P_d[j], 1.0)
                P_s_cal = dot(T_sd, P_d_homo)[0:3]
                # -- Project 3D world point to image plane
                Q_s_cal = dot(K_s, P_s_cal)
                # -- Normalize projected image point
                Q_s_cal[0] = Q_s_cal[0] / Q_s_cal[2]
                Q_s_cal[1] = Q_s_cal[1] / Q_s_cal[2]
                Q_s_cal = Q_s_cal[:2]
                # -- Calculate reprojection error
                err_s[(j * 2):(j * 2 + 2)] = Q_s[j] - Q_s_cal

            # Calculate reprojection error in the dynamic camera
            nb_P_s_corners = len(P_s)
            err_d = np.zeros(nb_P_s_corners * 2)
            for j in range(nb_P_s_corners):
                # -- Transform 3D world point from dynamic to static camera
                P_s_homo = np.append(P_s[j], 1.0)
                P_d_cal = dot(np.linalg.inv(T_sd), P_s_homo)[0:3]
                # -- Project 3D world point to image plane
                Q_d_cal = dot(K_d, P_d_cal)
                # -- Normalize projected image point
                Q_d_cal[0] = Q_d_cal[0] / Q_d_cal[2]
                Q_d_cal[1] = Q_d_cal[1] / Q_d_cal[2]
                Q_d_cal = Q_d_cal[:2]
                # -- Calculate reprojection error
                err_d[(j * 2):(j * 2 + 2)] = Q_d[j] - Q_d_cal

            # Stack residuals
            residuals.append(np.block([err_s, err_d]))

        return np.array(residuals).reshape((-1))

    def optimize(self):
        """ Optimize Gimbal Extrinsics """
        # Setup
        x, Z = self.setup_problem()
        K_s = self.data.cam0_intrinsics.K_new
        K_d = self.data.cam1_intrinsics.K_new
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
