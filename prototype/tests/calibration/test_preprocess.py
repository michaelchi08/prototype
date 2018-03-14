import unittest
from os.path import join

import cv2

import prototype.tests as test
from prototype.calibration.chessboard import Chessboard
from prototype.calibration.camera_intrinsics import CameraIntrinsics
from prototype.calibration.preprocess import PreprocessData


class PreprocessDataTest(unittest.TestCase):
    def test_ideal2pixels(self):
        # Setup
        self.data_path = join(test.TEST_DATA_PATH, "calib_data")
        images_dir = join(self.data_path, "gimbal_camera")
        intrinsics_file = join(self.data_path, "static_camera.yaml")
        intrinsics = CameraIntrinsics(intrinsics_file)
        chessboard = Chessboard(nb_rows=6, nb_cols=7, square_size=0.29)
        self.data = PreprocessData("IMAGES",
                                   images_dir=images_dir,
                                   chessboard=chessboard,
                                   intrinsics=intrinsics)
        # Load test image
        image_path = join(self.data_path, "static_camera", "img_0.jpg")
        image = cv2.imread(image_path)

        # cv2.imshow("Image", image)
        # cv2.waitKey()

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
        self.data = PreprocessData("IMAGES",
                                   images_dir=images_dir,
                                   chessboard=chessboard,
                                   intrinsics=intrinsics)

        # load data
        self.data.preprocess()
        self.assertTrue(len(self.data.images) > 0)
        self.assertTrue(len(self.data.images_ud) > 0)

    # def test_load_preprocessed(self):
    #     self.data_path = "/home/chutsu/Dropbox/calib_data"
    #     intrinsics_file = join(self.data_path, "static_camera.yaml")
    #     intrinsics = CameraIntrinsics(intrinsics_file)
    #     self.data = PreprocessData("PREPROCESSED",
    #                                data_path=join(self.data_path, "cam0"),
    #                                intrinsics=intrinsics)
    #     self.data.load()
