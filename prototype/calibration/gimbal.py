from os.path import join
from math import pi

import cv2
import yaml
import numpy as np
from numpy import dot

from prototype.utils.filesystem import walkdir
from prototype.models.gimbal import GimbalModel
from prototype.calibration.chessboard import Chessboard


class CameraIntrinsics:
    """ Camera intrinsics

    Attributes
    ----------
    cam_id : str
        Camera ID
    camera_model : str
        Camera model
    distortion_model : str
        Distortion model
    distortion_coeffs : str
        Distortion coefficients
    intrinsics : np.array
        Camera intrinsics
    resolution : np.array
        Camera resolution

    """
    def __init__(self, cam_id, filepath):
        self.cam_id = None
        self.camera_model = None
        self.distortion_model = None
        self.distortion_coeffs = None
        self.intrinsics = None
        self.resolution = None

        self.K_new = None

        self.load(cam_id, filepath)

    def load(self, cam_id, filepath):
        """ Load camera intrinsics

        `filepath` is expected to point towards a yaml file produced by
        Kalibr's camera calibration process, the output is expected to have the
        following format:

        [Kalibr]: https://github.com/ethz-asl/kalibr

        ```
        cam0:
            cam_overlaps: [1]
            camera_model: pinhole
            distortion_coeffs: [k1, k2, k3, k4]
            distortion_model: equidistant
            intrinsics: [fx, fy, cx, cy]
            resolution: [px, py]
            rostopic: "..."
        cam1:
            T_cn_cnm1:
            - [1, 0, 0, 0]
            - [0, 1, 0, 0]
            - [0, 0, 1, 0]
            - [0, 0, 0, 1]
            cam_overlaps: [0]
            camera_model: pinhole
            distortion_coeffs: [k1, k2, k3, k4]
            distortion_model: equidistant
            intrinsics: [fx, fy, cx, cy]
            resolution: [px, py]
            rostopic: "..."
        ```

        Parameters
        ----------
        cam_id : int
            Camera ID
        filepath : str
            Path to camera intrinsics file

        """
        intrinsics_file = open(filepath, "r")
        intrinsics_txt = intrinsics_file.read()
        intrinsics_file.close()
        intrinsics = yaml.load(intrinsics_txt)

        self.cam_id = "cam%d" % (cam_id)
        self.camera_model = intrinsics[self.cam_id]["camera_model"]
        self.distortion_model = intrinsics[self.cam_id]["distortion_model"]
        self.distortion_coeffs = intrinsics[self.cam_id]["distortion_coeffs"]
        self.intrinsics = intrinsics[self.cam_id]["intrinsics"]
        self.resolution = intrinsics[self.cam_id]["resolution"]

    def K(self):
        """ Form camera intrinsics matrix K """
        fx, fy, cx, cy = self.intrinsics
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        return K

    def D(self):
        """ Form distortion coefficients vector D """
        return np.array(self.distortion_coeffs)

    def undistort_points(self, points):
        """ Undistort points

        Parameters
        ----------
        points : np.array
            Points to undistort in pixel coordinates

        Return
        ----------
        points : np.array
            Undistorted points in ideal coordinates

        """
        # Get distortion model and form camera intrinsics matrix K
        distortion_coeffs = np.array(self.distortion_coeffs)
        K = self.K()

        # Undistort points
        if self.distortion_model == "radtan":
            # Distortion coefficients (k1, k2, r1, r2)
            points = cv2.undistortPoints(points, K, distortion_coeffs)
        elif self.distortion_model == "equidistant":
            # Distortion coefficients (k1, k2, k3, k4)
            points = cv2.fisheye.undistortPoints(points, K, distortion_coeffs)

        return points

    def undistort_image(self, image):
        # Get distortion model and form camera intrinsics matrix K
        distortion_coeffs = np.array(self.distortion_coeffs)
        K = self.K()

        # Undistort points
        if self.distortion_model == "radtan":
            D = distortion_coeffs  # (k1, k2, r1, r2)
            image = cv2.undistort(image, K, distortion_coeffs)

        elif self.distortion_model == "equidistant":
            D = distortion_coeffs  # (k1, k2, k3, k4)
            img_size = (image.shape[1], image.shape[0])
            R = np.eye(3)
            balance = 0.0

            K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K,
                D,
                img_size,
                R,
                balance=balance
            )
            undistorted_image = cv2.fisheye.undistortImage(image,
                                                           K,
                                                           D,
                                                           None,
                                                           K_new)

        self.K_new = K_new
        return undistorted_image, K_new

    def __str__(self):
        """ CameraIntrinsics to string """
        s = "cam_id: " + self.cam_id + "\n"
        s += "camera_model: " + self.camera_model + "\n"
        s += "distortion_model: " + self.distortion_model + "\n"
        s += "distortion_coeffs: " + str(self.distortion_coeffs) + "\n"
        s += "intrinsics: " + str(self.intrinsics) + "\n"
        s += "resolution: " + str(self.resolution)
        return s


class GimbalCalibData:
    """ Gimbal calibration data

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

        self.object_points = self.create_object_points(self.chessboard)
        self.cam0_corners2d = []
        self.cam1_corners2d = []
        self.cam0_corners3d = []
        self.cam1_corners3d = []
        self.imu_data = []

        self.cam0_intrinsics = None
        self.cam1_intrinsics = None

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
        self.cam0_intrinsics = CameraIntrinsics(0, intrinsics_fpath)
        self.cam1_intrinsics = CameraIntrinsics(1, intrinsics_fpath)

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
            fx, fy, cx, cy = self.cam0_intrinsics.intrinsics
        elif cam_id == 1:
            fx, fy, cx, cy = self.cam1_intrinsics.intrinsics

        # Convert ideal points to pixel coordinates
        pixels = []
        nb_points = len(points)
        for p in points.reshape((nb_points, 2)):
            px = (p[0] * fx) + cx
            py = (p[1] * fy) + cy
            pixels.append([px, py])

        return np.array([pixels])

    def create_object_points(self, chessboard):
        # Hard-coding object points - assuming chessboard is origin by
        # setting chessboard in the x-y plane (where z = 0).
        object_points = []
        for i in range(chessboard.nb_rows):
            for j in range(chessboard.nb_cols):
                pt = [j * chessboard.square_size,
                      i * chessboard.square_size,
                      0.0]
                object_points.append(pt)
        object_points = np.array(object_points)
        return object_points

    def solvepnp_chessboard(self, chessboard, corners, K, D=np.zeros(4)):
        # Calculate transformation matrix
        retval, rvec, tvec = cv2.solvePnP(self.object_points,
                                          corners,
                                          K,
                                          D,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
        if retval is False:
            raise RuntimeError("solvePnP failed!")

        # Convert rotation vector to matrix
        R = np.zeros((3, 3))
        cv2.Rodrigues(rvec, R)

        # Form transformation matrix
        T = np.array([[R[0][0], R[0][1], R[0][2], tvec[0]],
                      [R[1][0], R[1][1], R[1][2], tvec[1]],
                      [R[2][0], R[2][1], R[2][2], tvec[2]],
                      [0.0, 0.0, 0.0, 1.0]])

        return T, rvec, tvec

    def draw_coord_frame(self, chessboard, img, corners, K, D=np.zeros(4)):
        # Coordinate frame
        axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)
        axis = self.chessboard.square_size * axis  # Scale chessboard sq size

        # Solve PnP and project coordinate frame to image plane
        T, rvec, tvec = self.solvepnp_chessboard(chessboard, corners, K, D)
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
        cam0_img_ud = self.draw_corners(cam0_img_ud, pixels0_ud)
        cam0_img = self.draw_coord_frame(self.chessboard,
                                         cam0_img,
                                         corners0,
                                         self.cam0_intrinsics.K())
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
                                         self.cam1_intrinsics.K())
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
        assert self.object_points is not None

        nb_obj_pts = self.object_points.shape[0]
        obj_pts_homo = np.block([self.object_points, np.ones((nb_obj_pts, 1))])
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
            corners0_ud = self.cam0_intrinsics.undistort_points(corners0)
            result = self.cam0_intrinsics.undistort_image(cam0_img)
            cam0_img_ud, K0_new = result
            pixels0_ud = self.ideal2pixel(0, corners0_ud, K0_new)
            pixels0_ud = self.ideal2pixel(0, corners0_ud, K0_new)

            # Undistort corners in camera 1
            corners1_ud = self.cam1_intrinsics.undistort_points(corners1)
            result = self.cam1_intrinsics.undistort_image(cam1_img)
            cam1_img_ud, K1_new = result
            pixels1_ud = self.ideal2pixel(1, corners1_ud, K1_new)

            # Visually inspect images
            if imshow:
                self.visual_inspect_images(cam0_img, cam0_img_ud, K0_new,
                                           corners0, pixels0_ud,
                                           cam1_img, cam1_img_ud, K1_new,
                                           corners1, pixels1_ud)
                if cv2.waitKey(0) == 113:
                    return False

            # Calculate camera to chessboard transform
            T_c0_cb, _, _ = self.solvepnp_chessboard(
                self.chessboard,
                pixels0_ud,
                self.cam0_intrinsics.K_new
            )
            T_c1_cb, _, _ = self.solvepnp_chessboard(
                self.chessboard,
                pixels1_ud,
                self.cam1_intrinsics.K_new
            )
            P_c0 = self.calc_chessboard_corner_positions(T_c0_cb)
            P_c1 = self.calc_chessboard_corner_positions(T_c1_cb)

            # Append to calibration data
            self.cam0_corners2d.append(pixels0_ud[0])
            self.cam1_corners2d.append(pixels1_ud[0])
            self.cam0_corners3d.append(P_c0)
            self.cam1_corners3d.append(P_c1)
            self.imu_data.append(imu_data[i, :])

        self.cam0_corners2d = np.array(self.cam0_corners2d)
        self.cam1_corners2d = np.array(self.cam1_corners2d)
        self.imu_data = np.array(self.imu_data)
        return True


class GimbalCalibration:
    """ Gimbal Calibration

    Attributes
    ----------
    gimbal_model : GimbalModel
        Gimbal model
    data : GimbalCalibData
        Calibration data

    """
    def __init__(self, **kwargs):
        self.gimbal_model = GimbalModel()
        self.data = GimbalCalibData(**kwargs)
        self.data.load()
        # self.data.load(imshow=True)

    def setup_problem(self):
        """ Setup the calibration optimization problem

        Returns
        -------
        x : np.array
            Vector of optimization parameters to be optimized

        """
        assert len(self.data.cam0_corners2d) == len(self.data.cam1_corners2d)
        assert len(self.data.cam0_corners3d) == len(self.data.cam0_corners3d)

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

        for i in range(int(K)):
            # Get joint angles
            Lambda1 = x[18] # Roll
            Lambda2 = x[18+1] # Pitch

            # Get measurements
            P_s, P_d, Q_s, Q_d = Z[i]

            # Calculate
            T_sd = self.gimbal_model.T_sd(tau_s, Lambda1, w1, Lambda2, w2, tau_d)

            for j in range(len(P_d)):
                P_d_homo = np.append(P_d[j], 1.0)
                print("P_s calculated: ", dot(T_sd, P_d_homo))
                print("P_s measured: ", P_s[j])

        # P_s_homo = np.append(P_s[0], 1.0)
        # print("P_d calculated: ", dot(np.linalg.inv(T_sd), P_s_homo)[0:3])
        # print("P_d measured: ", P_d[0])

        # Calculate reprojection error on static camera
        # -- Project 3D world point to image plane
        # p_s_homo = np.append(p_s, [1])
        # x_C = dot(K_d, dot(T_d_s, p_s_homo)[:3])
        # -- Normalize projected image point
        # x_C[0] = x_C[0] / x_C[2]
        # x_C[1] = x_C[1] / x_C[2]
        # x_C = x_C[:2]
        # -- Calculate residual error
        # residual = z_d - x_C
        # print("residual: ", residual)

        # Calculate reprojection error on dynamic camera
        # -- Project 3D world point to image plane
        # p_s_homo = np.append(p_s, [1])
        # x_C = dot(K_d, dot(T_d_s, p_s_homo)[:3])
        # -- Normalize projected image point
        # x_C[0] = x_C[0] / x_C[2]
        # x_C[1] = x_C[1] / x_C[2]
        # x_C = x_C[:2]
        # -- Calculate residual error
        # residual = z_d - x_C
        # print("residual: ", residual)
