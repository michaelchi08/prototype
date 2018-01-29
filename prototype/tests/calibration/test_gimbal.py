import unittest
from os.path import join

import cv2
import numpy as np
import matplotlib.pyplot as plt

import prototype.tests as test
from prototype.calibration.chessboard import Chessboard
from prototype.calibration.camera_intrinsics import CameraIntrinsics
from prototype.calibration.gimbal import ECData
from prototype.calibration.gimbal import GECDataLoader
from prototype.calibration.gimbal import GEC
from prototype.calibration.gimbal import GimbalDataGenerator


class ECDataTest(unittest.TestCase):
    def setUp(self):
        self.data_path = join(test.TEST_DATA_PATH, "calib_data")
        images_dir = join(self.data_path, "gimbal_camera")
        intrinsics_file = join(self.data_path, "static_camera.yaml")
        intrinsics = CameraIntrinsics(intrinsics_file)
        chessboard = Chessboard(nb_rows=6, nb_cols=7, square_size=0.29)
        self.data = ECData(images_dir, intrinsics, chessboard)

    def test_ideal2pixels(self):
        # Load test image
        image_path = join(self.data_path, "static_camera", "img_0.jpg")
        image = cv2.imread(image_path)
        cv2.waitKey(0)

        # Detect chessboard corners
        chessboard = Chessboard(nb_rows=6, nb_cols=7, square_size=0.29)
        corners = chessboard.find_corners(image)

        # Convert points from ideal to pixel coordinates
        corners_ud = self.data.intrinsics.undistort_points(corners)
        self.data.intrinsics.undistort_image(image)
        K_new = self.data.intrinsics.K_new
        self.data.ideal2pixel(corners_ud, K_new)

    def test_load(self):
        self.data.load()
        self.assertTrue(len(self.data.images) > 0)
        self.assertTrue(len(self.data.images_ud) > 0)


class GECDataLoaderTest(unittest.TestCase):
    def setUp(self):
        self.data_path = join(test.TEST_DATA_PATH, "calib_data")
        self.loader = GECDataLoader(
            data_path=self.data_path,
            image_dirs=["static_camera", "gimbal_camera"],
            intrinsic_files=["static_camera.yaml", "gimbal_camera.yaml"],
            imu_file="imu.dat",
            nb_rows=6,
            nb_cols=7,
            square_size=0.29
        )

    def test_load_imu_data(self):
        imu_data = self.loader.load_imu_data()
        self.assertEqual(5, imu_data.shape[0])
        self.assertEqual(3, imu_data.shape[1])

    def test_load(self):
        self.loader.load()
    #     self.assertTrue(retval)


class GECTest(unittest.TestCase):
    def setUp(self):
        self.data_path = join(test.TEST_DATA_PATH, "calib_data")
        self.calib = GEC(
            data_path=self.data_path,
            image_dirs=["static_camera", "gimbal_camera"],
            intrinsic_files=["static_camera.yaml", "gimbal_camera.yaml"],
            imu_file="imu.dat",
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

    # def test_optimize(self):
    #     self.data_path = "/home/chutsu/Dropbox/calib_data/extrinsics"
    #     calib = GEC(
    #         data_path=self.data_path,
    #         image_dirs=["static_camera", "gimbal_camera"],
    #         intrinsic_files=["static_camera.yaml", "gimbal_camera.yaml"],
    #         imu_file="gimbal_joint.dat",
    #         nb_rows=6,
    #         nb_cols=7,
    #         square_size=0.0285
    #     )
    #     calib.optimize()


class GimbalDataGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.data_path = join(test.TEST_DATA_PATH, "calib_data2")
        self.intrinsics_file = join(self.data_path, "camera.yaml")
        self.data = GimbalDataGenerator(self.intrinsics_file)

    # def test_sandbox(self):
    #     data.plot()

    def test_generate(self):
        ec_data, imu_data = self.data.generate()
        gec = GEC(sim_mode=True, ec_data=ec_data, imu_data=imu_data)
        gec.optimize()
