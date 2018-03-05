from os.path import join

import cv2
import numpy as np

from prototype.calibration.chessboard import Chessboard
from prototype.calibration.camera_intrinsics import CameraIntrinsics
from prototype.calibration.preprocess import PreprocessData


class DataLoader:
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
        self.preprocessed = kwargs.get("preprocessed", False)
        self.inspect_data = kwargs.get("inspect_data", False)
        self.joint_file = kwargs["joint_file"]

        if self.preprocessed is False:
            self.image_dirs = kwargs["image_dirs"]
            self.intrinsic_files = kwargs["intrinsic_files"]
            self.chessboard = Chessboard(**kwargs)
        else:
            self.data_dirs = kwargs["data_dirs"]
            self.intrinsic_files = kwargs["intrinsic_files"]

    def load_joint_data(self):
        """ Load joint data

        Parameters
        ----------
        joint_fpath : str
            Joint data file path

        Returns
        -------
        joint_data : np.array
            IMU data

        """
        joint_file = open(join(self.data_path, self.joint_file), "r")
        joint_data = np.loadtxt(joint_file, delimiter=",")
        joint_file.close()
        return joint_data

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

    def preprocess_images(self):
        """ Preprocess images """
        # Load camera data
        nb_cameras = len(self.image_dirs)
        data = []
        for i in range(nb_cameras):
            image_dir = join(self.data_path, self.image_dirs[i])
            intrinsics_file = join(self.data_path, self.intrinsic_files[i])
            intrinsics = CameraIntrinsics(intrinsics_file)
            data_entry = PreprocessData("IMAGES",
                                        images_dir=image_dir,
                                        chessboard=self.chessboard,
                                        intrinsics=intrinsics)
            data_entry.load()
            data.append(data_entry)

        # Inspect data
        self.check_nb_images(data)
        if self.inspect_data is False:
            return data

        nb_images = len(data[0].images)
        for i in range(nb_images):
            viz = data[0].get_viz(i)
            for n in range(1, nb_cameras):
                viz = np.vstack((viz, data[n].get_viz(i)))
            cv2.imshow("Image", viz)
            cv2.waitKey(0)

        return data

    def filter_common_observations(self, i, data):
        cam0_idx = 0
        cam1_idx = 0

        P_s = []
        P_d = []
        Q_s = []
        Q_d = []

        # Find common target points and store the
        # respective points in 3d and 2d
        for pt_a in data[0].target_points[i]:
            for pt_b in data[1].target_points[i]:
                if np.array_equal(pt_a, pt_b):
                    # Corners 3d observed in both the static and dynamic cam
                    P_s.append(data[0].corners3d_ud[i][cam0_idx])
                    P_d.append(data[1].corners3d_ud[i][cam1_idx])

                    # Corners 2d observed in both the static and dynamic cam
                    Q_s.append(data[0].corners2d_ud[i][cam0_idx])
                    Q_d.append(data[1].corners2d_ud[i][cam1_idx])
                    break

                else:
                    cam1_idx += 1

            cam0_idx += 1
            cam1_idx = 0

        P_s = np.array(P_s)
        P_d = np.array(P_d)
        Q_s = np.array(Q_s)
        Q_d = np.array(Q_d)

        return [P_s, P_d, Q_s, Q_d]

    def load_preprocessed(self):
        # Load data from each camera
        data = []
        for i in range(len(self.data_dirs)):
            intrinsics_path = join(self.data_path, self.intrinsic_files[i])
            intrinsics = CameraIntrinsics(intrinsics_path)
            data_path = join(self.data_path, self.data_dirs[i])
            data_entry = PreprocessData("PREPROCESSED",
                                        data_path=data_path,
                                        intrinsics=intrinsics)
            data_entry.load()
            data.append(data_entry)

        # Find common measurements between cameras
        Z = []
        nb_measurements = len(data[0].target_points)
        # -- Iterate through measurement sets
        for i in range(nb_measurements):
            Z_i = self.filter_common_observations(i, data)
            Z.append(Z_i)

        # Camera intrinsics
        intrinsics_path = join(self.data_path, self.intrinsic_files[0])
        # K_s = CameraIntrinsics(intrinsics_path).K()
        C_s_intrinsics = CameraIntrinsics(intrinsics_path)
        K_s = C_s_intrinsics.calc_Knew()
        D_s = C_s_intrinsics.distortion_coeffs

        intrinsics_path = join(self.data_path, self.intrinsic_files[1])
        # K_d = CameraIntrinsics(intrinsics_path).K()
        C_d_intrinsics = CameraIntrinsics(intrinsics_path)
        K_d = C_d_intrinsics.calc_Knew()
        D_d = C_d_intrinsics.distortion_coeffs

        return Z, K_s, K_d, D_s, D_d

    def load(self):
        """ Load calibration data """
        # Load joint data
        joint_data = self.load_joint_data()

        # Load data
        if self.preprocessed is False:
            data = self.preprocess_images()
            K = len(data[0].corners2d_ud)

            # Setup measurement sets
            Z = []
            for i in range(K):
                # Corners 3d observed in both the static and dynamic cam
                P_s = data[0].corners3d_ud[i]
                P_d = data[1].corners3d_ud[i]
                # Corners 2d observed in both the static and dynamic cam
                Q_s = data[0].corners2d_ud[i]
                Q_d = data[1].corners2d_ud[i]
                Z_i = [P_s, P_d, Q_s, Q_d]
                Z.append(Z_i)
            K_s = data[0].intrinsics.K_new
            K_d = data[1].intrinsics.K_new
            D_s = data[0].intrinsics.distortion_coeffs
            D_d = data[1].intrinsics.distortion_coeffs
            return Z, K_s, K_d, D_s, D_d, joint_data

        else:
            Z, K_s, K_d, D_s, D_d = self.load_preprocessed()
            return Z, K_s, K_d, D_s, D_d, joint_data
