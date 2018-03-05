import unittest
from os.path import join

import prototype.tests as test
from prototype.calibration.dataset import GimbalDataGenerator
from prototype.calibration.calibration import GimbalCalibrator


class GimbalDataGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.data_path = join(test.TEST_DATA_PATH, "calib_data2")
        self.intrinsics_file = join(self.data_path, "camera.yaml")
        self.data = GimbalDataGenerator(self.intrinsics_file)

    def test_generate(self):
        Z, K_s, K_d, D_s, D_d, joint_data = self.data.generate()
        # self.data.plot()

        calibrator = GimbalCalibrator(sim_mode=True,
                                      Z=Z,
                                      K_s=K_s,
                                      K_d=K_d,
                                      D_s=D_s,
                                      D_d=D_d,
                                      joint_data=joint_data)

        calibrator.gimbal_model.tau_s[0] += 0.1
        calibrator.gimbal_model.tau_s[1] += 0.2
        calibrator.gimbal_model.tau_s[2] += 0.2

        calibrator.gimbal_model.tau_d[0] += 0.1
        calibrator.gimbal_model.tau_d[1] += 0.2
        calibrator.gimbal_model.tau_d[2] += 0.2

        # x, Z, K_s, K_d, D_s, D_d = calibrator.setup_problem()
        # args = [Z, K_s, K_d, D_s, D_d]
        # result = calibrator.reprojection_error(x, *args)
        # print(max(result))

        calibrator.optimize()
