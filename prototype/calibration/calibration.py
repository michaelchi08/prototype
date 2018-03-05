import numpy as np
from numpy import dot
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from prototype.models.gimbal import GimbalModel
from prototype.viz.plot_gimbal import PlotGimbal
from prototype.calibration.loader import DataLoader
from prototype.vision.camera.distortion_model import project_pinhole_equi


class GimbalCalibrator:
    """ Gimbal Extrinsics Calibrator

    Attributes
    ----------
    gimbal_model : GimbalModel
        Gimbal model
    data : GECDataLoader
        Calibration data

    """
    def __init__(self, **kwargs):
        self.gimbal_model = kwargs.get("gimbal_model", GimbalModel())

        if kwargs.get("sim_mode", False):
            # Load sim data
            self.Z = kwargs["Z"]
            self.K_s = kwargs["K_s"]
            self.K_d = kwargs["K_d"]
            self.D_s = kwargs["D_s"]
            self.D_d = kwargs["D_d"]
            self.joint_data = kwargs["joint_data"]
            self.K = len(self.Z)

        else:
            # Load data
            self.loader = DataLoader(**kwargs)
            # -- Measurement set and joint data
            data = self.loader.load()
            self.Z, self.K_s, self.K_d, self.D_s, self.D_d, self.joint_data = data
            # -- Number of measurement set
            self.K = len(self.Z)

    def setup_problem(self):
        """ Setup the calibration optimization problem

        Returns
        -------
        x : np.array
            Vector of optimization parameters to be optimized

        """
        print("Setting up optimization problem ...")

        # Parameters to be optimized
        x_size = 6 + 5 + 3 + self.K * 2
        x = np.zeros(x_size)
        # -- tau_s
        x[0] = self.gimbal_model.tau_s[0]
        x[1] = self.gimbal_model.tau_s[1]
        x[2] = self.gimbal_model.tau_s[2]
        x[3] = self.gimbal_model.tau_s[3]
        x[4] = self.gimbal_model.tau_s[4]
        x[5] = self.gimbal_model.tau_s[5]
        # -- tau_d
        x[6] = self.gimbal_model.tau_d[0]
        x[7] = self.gimbal_model.tau_d[1]
        x[8] = self.gimbal_model.tau_d[2]
        x[9] = self.gimbal_model.tau_d[4]
        x[10] = self.gimbal_model.tau_d[5]
        # -- alpha, a, d
        x[11] = self.gimbal_model.link[1]
        x[12] = self.gimbal_model.link[2]
        x[13] = self.gimbal_model.link[3]
        # -- Joint angles
        x[14:] = self.joint_data[:, 0:2].ravel()

        return x, self.Z, self.K_s, self.K_d, self.D_s, self.D_d

    def reprojection_error(self, x, *args):
        """Reprojection Error

        Parameters
        ----------
        x : np.array
            Parameters to be optimized
        args : tuple of (Z, K_s, K_d)
            Z: list of measurement sets
            K_s: np.array static camera intrinsics matrix K
            K_d: np.array dynamic camera intrinsics matrix K

        Returns
        -------
        residual : np.array
            Reprojection error

        """
        # Map the optimization params back into the transforms
        # -- tau_s
        tau_s = x[0:6]
        # -- tau_d
        tau_d_tx = x[6]
        tau_d_ty = x[7]
        tau_d_tz = x[8]
        tau_d_pitch = x[9]
        tau_d_yaw = x[10]
        # -- alpha, a, d
        alpha, a, d = x[11:14]
        # -- Joint angles
        roll_angles = []
        pitch_angles = []
        for i in range(self.K):
            roll_angles.append(x[14 + (2 * i)])
            pitch_angles.append(x[14 + (2 * i) + 1])

        # Set gimbal model
        self.gimbal_model.tau_s = tau_s
        self.gimbal_model.tau_d = [tau_d_tx, tau_d_ty, tau_d_tz,
                                   None, tau_d_pitch, tau_d_yaw]
        self.gimbal_model.link = [None, alpha, a, d]

        # Loop through all measurement sets
        Z, K_s, K_d, D_s, D_d = args
        residuals = []
        for k in range(int(self.K)):
            # Get the k-th measurements
            P_s, P_d, Q_s, Q_d = Z[k]

            # Get joint angles
            roll = roll_angles[k]
            pitch = pitch_angles[k]
            self.gimbal_model.set_attitude([roll, pitch])

            # Get static to dynamic camera transform
            T_sd = self.gimbal_model.calc_transforms()[2]

            # Calculate reprojection error in the static camera
            nb_P_d_corners = len(P_d)
            err_s = np.zeros(nb_P_d_corners * 2)
            for i in range(nb_P_d_corners):
                # -- Transform 3D world point from dynamic to static camera
                P_d_homo = np.append(P_d[i], 1.0)
                P_s_cal = dot(T_sd, P_d_homo)[0:3]
                # -- Project 3D world point to image plane
                Q_s_cal = project_pinhole_equi(P_s_cal, K_s, D_s)
                # -- Calculate reprojection error
                err_s[(i * 2):(i * 2 + 2)] = Q_s[i] - Q_s_cal

            # Calculate reprojection error in the dynamic camera
            nb_P_s_corners = len(P_s)
            err_d = np.zeros(nb_P_s_corners * 2)
            for i in range(nb_P_s_corners):
                # -- Transform 3D world point from dynamic to static camera
                P_s_homo = np.append(P_s[i], 1.0)
                P_d_cal = dot(np.linalg.inv(T_sd), P_s_homo)[0:3]
                # -- Project 3D world point to image plane
                Q_d_cal = project_pinhole_equi(P_d_cal, K_d, D_d)
                # -- Calculate reprojection error
                err_d[(i * 2):(i * 2 + 2)] = Q_d[i] - Q_d_cal

            # Stack residuals
            residuals.append(np.block([err_s, err_d]))

        result = np.array(residuals).reshape((-1))
        result = np.hstack(result)

        return result

    def optimize(self):
        """ Optimize Gimbal Extrinsics """
        # Setup
        x, Z, K_s, K_d, D_s, D_d = self.setup_problem()
        args = [Z, K_s, K_d, D_s, D_d]

        # Optimize
        print("Optimizing!")
        print("This can take a while...")
        result = least_squares(self.reprojection_error,
                               x,
                               args=args,
                               verbose=2)

        # Parse results
        tau_s = result.x[0:6]

        tau_d_tx = result.x[6]
        tau_d_ty = result.x[7]
        tau_d_tz = result.x[8]
        tau_d_roll = 0.0
        tau_d_pitch = result.x[9]
        tau_d_yaw = result.x[10]
        tau_d = [tau_d_tx, tau_d_ty, tau_d_tz,
                 tau_d_roll, tau_d_pitch, tau_d_yaw]

        alpha, a, d = result.x[11:14]

        self.gimbal_model.tau_s = tau_s
        self.gimbal_model.tau_d = tau_d
        self.gimbal_model.link = [0.0, alpha, a, d]

        print("Results:")
        print("---------------------------------")
        print("tau_s: ", self.gimbal_model.tau_s)
        print("tau_d: ", self.gimbal_model.tau_d)
        print("w1: ", self.gimbal_model.link)

        # Plot gimbal
        self.gimbal_model.set_attitude([0.0, 0.0])
        plot_gimbal = PlotGimbal(gimbal=self.gimbal_model)
        plot_gimbal.plot()
        plt.show()
