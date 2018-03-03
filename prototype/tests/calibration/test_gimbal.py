import unittest
from os.path import join
from math import pi

import cv2
import numpy as np
import matplotlib.pyplot as plt

import prototype.tests as test
from prototype.utils.utils import deg2rad
from prototype.models.gimbal import GimbalModel
from prototype.viz.plot_gimbal import PlotGimbal
from prototype.calibration.chessboard import Chessboard
from prototype.calibration.camera_intrinsics import CameraIntrinsics
from prototype.calibration.gimbal import ECData
from prototype.calibration.gimbal import GECDataLoader
from prototype.calibration.gimbal import GEC
from prototype.calibration.gimbal import GimbalDataGenerator


class ECDataTest(unittest.TestCase):
    def test_ideal2pixels(self):
        # Setup
        self.data_path = join(test.TEST_DATA_PATH, "calib_data")
        images_dir = join(self.data_path, "gimbal_camera")
        intrinsics_file = join(self.data_path, "static_camera.yaml")
        intrinsics = CameraIntrinsics(intrinsics_file)
        chessboard = Chessboard(nb_rows=6, nb_cols=7, square_size=0.29)
        self.data = ECData("IMAGES",
                           images_dir=images_dir,
                           chessboard=chessboard,
                           intrinsics=intrinsics)
        # Load test image
        image_path = join(self.data_path, "static_camera", "img_0.jpg")
        image = cv2.imread(image_path)

        cv2.imshow("Image", image)
        cv2.waitKey()

        # Detect chessboard corners
        chessboard = Chessboard(nb_rows=6, nb_cols=7, square_size=0.29)
        corners = chessboard.find_corners(image)

        # Convert points from ideal to pixel coordinates
        corners_ud = self.data.intrinsics.undistort_points(corners)
        self.data.intrinsics.undistort_image(image)
        K_new = self.data.intrinsics.K_new
        self.data.ideal2pixel(corners_ud, K_new)

    def test_load(self):
        # Setup
        self.data_path = join(test.TEST_DATA_PATH, "calib_data")
        images_dir = join(self.data_path, "gimbal_camera")
        intrinsics_file = join(self.data_path, "static_camera.yaml")
        intrinsics = CameraIntrinsics(intrinsics_file)
        chessboard = Chessboard(nb_rows=6, nb_cols=7, square_size=0.29)
        self.data = ECData("IMAGES",
                           images_dir=images_dir,
                           chessboard=chessboard,
                           intrinsics=intrinsics)

        # load data
        self.data.load()
        self.assertTrue(len(self.data.images) > 0)
        self.assertTrue(len(self.data.images_ud) > 0)

    def test_load_preprocessed(self):
        self.data_path = "/home/chutsu/Dropbox/calib_data"
        intrinsics_file = join(self.data_path, "static_camera.yaml")
        intrinsics = CameraIntrinsics(intrinsics_file)
        self.data = ECData("PREPROCESSED",
                           data_path=join(self.data_path, "cam0"),
                           intrinsics=intrinsics)
        self.data.load()


class GECDataLoaderTest(unittest.TestCase):
    def setUp(self):
        self.data_path = join(test.TEST_DATA_PATH, "calib_data")

    def test_load_imu_data(self):
        loader = GECDataLoader(
            data_path=self.data_path,
            image_dirs=["static_camera", "gimbal_camera"],
            intrinsic_files=["static_camera.yaml", "gimbal_camera.yaml"],
            joint_file="joint.csv",
            nb_rows=6,
            nb_cols=7,
            square_size=0.29
        )

        joint_data = loader.load_joint_data()
        self.assertEqual(5, joint_data.shape[0])
        self.assertEqual(3, joint_data.shape[1])

    def test_load(self):
        loader = GECDataLoader(
            data_path=self.data_path,
            inspect_data=True,
            image_dirs=["static_camera", "gimbal_camera"],
            intrinsic_files=["static_camera.yaml", "gimbal_camera.yaml"],
            joint_file="joint.csv",
            nb_rows=6,
            nb_cols=7,
            square_size=0.29
        )
        loader.load()

    def test_load_preprocessed(self):
        data_path = "/home/chutsu/Dropbox/calib_data/"
        loader = GECDataLoader(
            preprocessed=True,
            data_path=data_path,
            data_dirs=["cam0", "cam1"],
            intrinsic_files=["static_camera.yaml", "gimbal_camera.yaml"],
            joint_file="joint.csv"
        )
        loader.load()


class GECTest(unittest.TestCase):
    def setUp(self):
        self.data_path = join(test.TEST_DATA_PATH, "calib_data")
        self.calib = GEC(
            data_path=self.data_path,
            image_dirs=["static_camera", "gimbal_camera"],
            intrinsic_files=["static_camera.yaml", "gimbal_camera.yaml"],
            joint_file="joint.csv",
            nb_rows=6,
            nb_cols=7,
            square_size=0.0285
        )

    def test_setup_problem(self):
        x, Z, K_s, K_d = self.calib.setup_problem()

        self.assertEqual((28, ), x.shape)
        self.assertEqual(5, len(Z))

    def test_reprojection_error(self):
        x, Z, K_s, K_d = self.calib.setup_problem()
        args = [Z, K_s, K_d]

        self.calib.reprojection_error(x, *args)

    def test_optimize(self):
        self.calib.optimize()

    def test_optimize_preprocessed(self):
        gimbal_model = GimbalModel(
            # tau_s=np.array([0.045, 0.08, -0.095, 0.0, 0.0, pi / 2.0]),
            # w1=np.array([pi / 2.0, 0.0, 0.045]),
            # tau_d=np.array([0.0, -0.03, 0.015, 0.0, pi / 2.0, -pi / 2.0]),
            # w2=np.array([pi, 0, 0])
            # tau_s=np.array([0.045, 0.08, -0.095, 0.0, 0.0, pi / 2.0]),
            # w1=np.array([-pi / 2.0, 0.0, 0.045]),
            # tau_d=np.array([0.0, 0.0, -0.015, 0.0, -pi / 2.0, -pi / 2.0]),
            # w2=np.array([pi, 0, 0])
            # tau_s=np.array([0.0, 0.0, 0.0, 0.0, 0.0, pi / 2.0]),
            # w1=np.array([-pi / 2.0, 0.0, 0.0]),
            # tau_d=np.array([0.0, 0.0, 0.0, 0.0, -pi / 2.0, -pi / 2.0]),
            # w2=np.array([pi, 0, 0])
            tau_s=np.array([0.045, 0.075, -0.085, 0.0, 0.0, pi / 2.0]),
            tau_d=np.array([0.0, 0.015, 0.0, 0.0, 0.0, -pi / 2.0]),
            w1=np.array([0.0, 0.0, 0.075]),
            w2=np.array([0.0, 0.0, 0.0])
        )
        # gimbal_model.set_attitude([deg2rad(0), deg2rad(0)])
        # plot_gimbal = PlotGimbal(gimbal=gimbal_model)
        # plot_gimbal.plot()
        # plt.show()

        data_path = "/home/chutsu/Dropbox/calib_data"
        calib = GEC(
            preprocessed=True,
            gimbal_model=gimbal_model,
            data_path=data_path,
            data_dirs=["cam0", "cam1"],
            intrinsic_files=["static_camera.yaml", "gimbal_camera.yaml"],
            joint_file="joint.csv"
        )
        calib.optimize()


class GimbalDataGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.data_path = join(test.TEST_DATA_PATH, "calib_data2")
        self.intrinsics_file = join(self.data_path, "camera.yaml")
        self.data = GimbalDataGenerator(self.intrinsics_file)

    # def test_sandbox(self):
    #     data.plot()

    def test_generate(self):
        ec_data, joint_data = self.data.generate()
        gec = GEC(sim_mode=True,
                  ec_data=ec_data,
                  joint_data=joint_data)
        gec.optimize()
