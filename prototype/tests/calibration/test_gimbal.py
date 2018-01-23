import unittest

# from prototype.calibration.gimbal import GimbalCalibration
from prototype.calibration.gimbal import GimbalCalibData


class GimbalCalibDataTest(unittest.TestCase):
    def test_constructor(self):
        data_path = "/home/chutsu/Dropbox/calib_data/extrinsics/"
        calib_data = GimbalCalibData(data_path=data_path,
                                     cam0_dir="static_camera",
                                     cam1_dir="gimbal_camera",
                                     intrinsics_filename="intrinsics.yaml",
                                     imu_filename="gimbal_joint.dat",
                                     nb_rows=6,
                                     nb_cols=7,
                                     square_size=0.29)
        calib_data.load()
