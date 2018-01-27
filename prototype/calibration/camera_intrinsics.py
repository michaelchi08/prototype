import cv2
import yaml
import numpy as np


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
    def __init__(self, filepath):
        self.camera_model = None
        self.distortion_model = None
        self.distortion_coeffs = None
        self.intrinsics = None
        self.resolution = None
        self.K_new = None

        self.load(filepath)

    def load(self, filepath):
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

        self.camera_model = intrinsics["camera_model"]
        self.distortion_model = intrinsics["distortion_model"]
        self.distortion_coeffs = intrinsics["distortion_coeffs"]
        self.intrinsics = intrinsics["intrinsics"]
        self.resolution = intrinsics["resolution"]

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
