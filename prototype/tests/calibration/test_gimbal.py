import unittest
from os.path import join

import cv2
import numpy as np

import prototype.tests as test
from prototype.calibration.chessboard import Chessboard
from prototype.calibration.gimbal import CameraIntrinsics
from prototype.calibration.gimbal import GimbalCalibData
from prototype.calibration.gimbal import GimbalCalibration


class CameraIntrinsicsTest(unittest.TestCase):
    def setUp(self):
        self.data_path = join(test.TEST_DATA_PATH, "calib_data")
        self.intrinsics_file = join(self.data_path, "intrinsics_equi.yaml")
        self.intrinsics = CameraIntrinsics(0, self.intrinsics_file)

    def test_load(self):
        self.assertEqual(self.intrinsics.cam_id, "cam0")
        self.assertEqual(self.intrinsics.camera_model, "pinhole")
        self.assertEqual(self.intrinsics.distortion_model, "equidistant")

        dist_actual = self.intrinsics.distortion_coeffs
        dist_expected = np.array([-0.03664954370736927,
                                  -0.01885374614074979,
                                  0.028828273806888644,
                                  -0.017664103517927136])
        self.assertTrue(np.array_equal(dist_expected, dist_actual))

        intrinsics_actual = self.intrinsics.intrinsics
        intrinsics_expected = np.array([359.8796971033865,
                                        361.4580570887365,
                                        341.87686392387604,
                                        255.91608889489478])
        self.assertTrue(np.array_equal(intrinsics_expected,
                                       intrinsics_actual))

        res_actual = self.intrinsics.resolution
        res_expected = np.array([640, 480])
        self.assertTrue(np.array_equal(res_expected, res_actual))

    def test_K(self):
        K = self.intrinsics.K()

        fx, fy, cx, cy = self.intrinsics.intrinsics
        self.assertEqual(fx, K[0, 0])
        self.assertEqual(fy, K[1, 1])
        self.assertEqual(cx, K[0, 2])
        self.assertEqual(cy, K[1, 2])

    def test_undistort_points(self):
        # Load corners
        img_path = join(self.data_path, "gimbal_camera", "img_0.jpg")
        img = cv2.imread(img_path)
        chessboard = Chessboard(nb_rows=6, nb_cols=7, square_size=0.29)
        corners = chessboard.find_corners(img)

        # Undistort points using equidistant distortion model
        points = self.intrinsics.undistort_points(corners)
        self.assertEqual(len(points), len(corners))
        self.assertEqual(points.shape[0], len(corners))
        self.assertEqual(points.shape[1], 1)
        self.assertEqual(points.shape[2], 2)

    def test_undistort_image(self):
        # Load image
        img_path = join(self.data_path, "gimbal_camera", "img_0.jpg")
        img = cv2.imread(img_path)

        # Undistort image using equidistant distortion model
        img, K_new = self.intrinsics.undistort_image(img)
        # cv2.imshow("Image", img)
        # cv2.waitKey(0)


# class GimbalCalibDataTest(unittest.TestCase):
#     def setUp(self):
#         self.data_path = join(test.TEST_DATA_PATH, "calib_data")
#         self.calib_data = GimbalCalibData(
#             data_path=self.data_path,
#             cam0_dir="static_camera",
#             cam1_dir="gimbal_camera",
#             intrinsics_filename="intrinsics_equi.yaml",
#             imu_filename="imu.dat",
#             nb_rows=6,
#             nb_cols=7,
#             square_size=0.29
#         )
#
#     def test_load_imu_data(self):
#         imu_fpath = join(self.data_path, self.calib_data.imu_filename)
#         imu_data = self.calib_data.load_imu_data(imu_fpath)
#         self.assertEqual(5, imu_data.shape[0])
#         self.assertEqual(3, imu_data.shape[1])
#
#     def test_load_cam_intrinsics(self):
#         intrinsics_fpath = join(self.data_path, "intrinsics_equi.yaml")
#         self.calib_data.load_cam_intrinsics(intrinsics_fpath)
#         self.assertTrue(self.calib_data.cam0_intrinsics is not None)
#         self.assertTrue(self.calib_data.cam1_intrinsics is not None)
#
#     def test_load(self):
#         retval = self.calib_data.load()
#         self.asserttrue(retval)


class GimbalCalibTest(unittest.TestCase):
    def setUp(self):
        self.data_path = join(test.TEST_DATA_PATH, "calib_data")
        self.calib = GimbalCalibration(
            data_path=self.data_path,
            cam0_dir="static_camera",
            cam1_dir="gimbal_camera",
            intrinsics_filename="intrinsics_equi.yaml",
            imu_filename="imu.dat",
            nb_rows=6,
            nb_cols=7,
            square_size=0.0285
        )

    def test_reprojection_error(self):
        # # Load image
        # img_path = join(self.data_path, "gimbal_camera", "img_0.jpg")
        # img = cv2.imread(img_path)

        # print(self.calib.data.object_points)
        # print(self.calib.data.cam0_intrinsics.K_new)
        # print(self.calib.data.cam0_T[0])

        # K = self.calib.data.cam0_intrinsics.K_new
        K = self.calib.data.cam0_intrinsics.K_new
        print("K: ", self.calib.data.cam0_intrinsics.K())
        print("K_new: ", self.calib.data.cam0_intrinsics.K_new)

        img_pt = self.calib.data.cam0_corners[0][0][0]
        print("img pt: ", img_pt)

        obj_pt = self.calib.data.object_points[0]
        print("obj_pt: ", obj_pt)

        obj_pt_homo = np.array([obj_pt[0], obj_pt[1], obj_pt[2], 1.0])
        T_C0_CB = self.calib.data.cam0_T[0]
        X_C = np.dot(T_C0_CB, obj_pt_homo)[:3]
        print("T_C0_CB:\n", T_C0_CB)
        print("X_C: ", X_C)

        self.calib.reprojection_error(K, img_pt, None, None, X_C)

        # cv2.imshow("Image", img)
        # cv2.waitKey(0)

        # K =
        # self.calib.reprojection_error(
        #
