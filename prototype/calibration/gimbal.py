from os.path import join

import cv2
import yaml
import numpy as np
from numpy import dot

from prototype.utils.euler import euler2rot
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

    cam0_corners : np.array
        Camera 0 image corners
    cam1_corners : np.array
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

        self.object_points = None
        self.cam0_corners = []
        self.cam1_corners = []
        self.cam0_T = []
        self.cam1_T = []
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

        return np.array([pixels]).reshape((nb_points, 1, 2))

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

    def solvepnp(self, chessboard, corners, K, D=np.zeros(4)):
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

        # Calculate transformation matrix
        retval, rvec, tvec = cv2.solvePnP(object_points,
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
        T, rvec, tvec = self.solvepnp(chessboard, corners, K, D)
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, D)

        # Draw coordinate frame
        corner = tuple(corners[0].ravel())
        corner = (int(corner[0]), int(corner[1]))
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 3)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 3)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 3)

        return img

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
        if len(cam0_files) != len(cam1_files):
            err = "Unequal num of images in [{0}] and [{1}]".format(
                cam0_path,
                cam1_path
            )
            raise RuntimeError(err)
        if len(cam0_files) != imu_data.shape[0]:
            err = "Unequal num of images [{0}] and imu data [{1}]".format(
                cam0_path,
                imu_fpath
            )
            raise RuntimeError(err)

        # Inspect images and prep calibration data
        nb_images = len(cam0_files)
        for i in range(nb_images):
            print("Inspecting image %d" % i)

            # Load images
            cam0_img = cv2.imread(cam0_files[i])
            cam1_img = cv2.imread(cam1_files[i])
            corners0 = self.chessboard.find_corners(cam0_img)
            corners1 = self.chessboard.find_corners(cam1_img)

            # Check if chessboard was detected in both cameras
            if corners0 is None or corners1 is None:
                continue  # Skip this for loop iteration

            # Undistort corners in camera 0
            corners0_ud = self.cam0_intrinsics.undistort_points(corners0)
            result = self.cam0_intrinsics.undistort_image(cam0_img)
            cam0_img_ud, K0_new = result
            pixels0_ud = self.ideal2pixel(0, corners0_ud, K0_new)

            # Undistort corners in camera 1
            corners1_ud = self.cam1_intrinsics.undistort_points(corners1)
            result = self.cam1_intrinsics.undistort_image(cam1_img)
            cam1_img_ud, K1_new = result
            pixels1_ud = self.ideal2pixel(1, corners1_ud, K1_new)

            # Visually inspect images
            if imshow:
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

                # Show camera 0 and camera 1
                # (detected points, undistored points)
                cam0_cam1_img = np.vstack((cam0_img_pair, cam1_img_pair))
                cv2.imshow("Inspect Images", cam0_cam1_img)
                if cv2.waitKey(0) == 113:
                    return False

            # Calculate camera to chessboard transform
            T_c0_cb, _, _ = self.solvepnp(self.chessboard, pixels0_ud,
                                          self.cam0_intrinsics.K_new)
            T_c1_cb, _, _ = self.solvepnp(self.chessboard, pixels0_ud,
                                          self.cam1_intrinsics.K_new)

            # Append to calibration data
            self.object_points = self.create_object_points(self.chessboard)
            self.cam0_corners.append(pixels0_ud)
            self.cam1_corners.append(pixels1_ud)
            self.cam0_T.append(T_c0_cb)
            self.cam1_T.append(T_c1_cb)
            self.imu_data.append(imu_data[i, :])

        return True


class GimbalCalibration:
    def __init__(self, **kwargs):
        self.gimbal_model = GimbalModel()
        self.data = GimbalCalibData(**kwargs)
        self.data.load()
        # self.data.load(imshow=True)

    def reprojection_error(self, K, image_point, rpy_C, t_C, X_C):
        """Reprojection Error

        Parameters
        ----------
        K : np.array of size 3x3
            Camera intrinsics
        image_point : np.array of size 3x1
            Image point
        rpy_C : np.array of size 3x1
            Roll, pitch yaw
        t_C : np.array of size 3x3
            Translation
        X : np.array of size 3x1
            World point

        Returns
        -------
        residual : float
            Reprojection error

        """
        # Convert euler to rotation matrix R
        # R_CB = euler2rot(rpy_C, 321)

        # Project 3D world point to image plane
        # x = dot(K, dot(R_CB, X_C - t_C))
        x = dot(K, X_C)

        # Normalize projected image point
        pt = np.array([0.0, 0.0])
        pt[0] = x[0] / x[2]
        pt[1] = x[1] / x[2]

        # Calculate residual error
        residual = image_point - pt
        print("residual: ", residual)

        return residual
