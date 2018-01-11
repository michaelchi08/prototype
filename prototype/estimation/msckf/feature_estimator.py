import numpy as np
from numpy import zeros
from numpy import dot
from scipy.optimize import least_squares

from prototype.utils.quaternion.jpl import C


class FeatureEstimator:
    """Feature Estimator

    Attributes
    ----------
    max_iter : int
        Maximum iterations

    Parameters
    ----------
    max_iter : int
        Maximum iterations (default: 30)

    """

    def triangulate(self, pt1, pt2, C_C0C1, t_C0_C0C1):
        """Triangulate feature observed from camera C0 and C1 and return the
        position relative to camera C0

        Parameters
        ----------
        pt1 : np.array - 2x1
            Feature point 1 in pixel coordinates
        pt2 : np.array - 2x1
            Feature point 2 in pixel coordinates
        C_C0C1 : np.array - 3x3
            Rotation matrix from frame C1 to C0
        t_C0_C0C1 : np.array - 3x3
            Translation vector from frame C0 to C1 expressed in C0

        Returns
        -------
        p_C0_f : np.array - 3x1
            Returns feature point expressed in C0 in camera coordinates

        """
        # Convert points to homogenous coordinates and normalize
        pt1 = np.block([[pt1], [1.0]])
        pt2 = np.block([[pt2], [1.0]])
        # pt1 = pt1 / np.linalg.norm(pt1)
        # pt2 = pt2 / np.linalg.norm(pt2)

        # Triangulate
        A = np.block([pt1, dot(-C_C0C1, pt2)])
        b = t_C0_C0C1
        result = np.linalg.lstsq(A, b)
        x, residual, rank, s = result
        p_C0_f = x[0] * pt1

        return p_C0_f, residual

    def initial_estimate(self, cam_model, track, track_cam_states):
        """Initial feature estimate

        Parameters
        ----------
        cam_model : CameraModel
            Camera model
        track : FeatureTrack
            Feature track
        track_cam_states : list of CameraState
            Camera states where feature track was observed
        debug :
            Debug mode (default: False)

        Returns
        -------
        p_C0_f : np.array - 3x1
            Initial estimated feature position expressed in first camera frame

        """
        # Calculate rotation and translation of first and last camera states
        # -- Get rotation and translation of camera 0 and camera 1
        C_C0G = C(track_cam_states[0].q_CG)
        C_C1G = C(track_cam_states[1].q_CG)
        p_G_C0 = track_cam_states[0].p_G
        p_G_C1 = track_cam_states[1].p_G
        # -- Calculate rotation and translation from camera 0 to camera 1
        C_C0C1 = dot(C_C0G, C_C1G.T)
        t_C1_C0C1 = dot(C_C0G, (p_G_C1 - p_G_C0))
        # -- Convert from pixel coordinates to image coordinates
        pt1 = cam_model.pixel2image(track.track[0].pt).reshape((2, 1))
        pt2 = cam_model.pixel2image(track.track[1].pt).reshape((2, 1))

        # Calculate initial estimate of 3D position
        p_C0_f, residual = self.triangulate(pt1, pt2, C_C0C1, t_C1_C0C1)

        return p_C0_f, residual

    def jacobian(self, x, *args):
        """Jacobian

        Parameters
        ----------
        cam_model : CameraModel
            Camera model
        track : FeatureTrack
            Feature track
        track_cam_states : list of CameraState
            Camera states where feature track was observed

        Returns
        -------
        J : np.array
            Jacobian

        """
        cam_model, track, track_cam_states = args

        N = len(track_cam_states)
        J = zeros((2 * N, 3))

        C_C0G = C(track_cam_states[0].q_CG)
        p_G_C0 = track_cam_states[0].p_G
        alpha, beta, rho = x.ravel()

        # Form the Jacobian
        for i in range(N):
            # Get camera current rotation and translation
            C_CiG = C(track_cam_states[i].q_CG)
            p_G_Ci = track_cam_states[i].p_G

            # Set camera 0 as origin, work out rotation and translation
            # of camera i relative to to camera 0
            C_CiC0 = dot(C_CiG, C_C0G.T)
            t_Ci_CiC0 = dot(C_CiG, (p_G_C0 - p_G_Ci))

            # Project estimated feature location to image plane
            h = dot(C_CiC0, np.array([[alpha], [beta], [1]])) + rho * t_Ci_CiC0  # noqa

            drdalpha = np.array([
                -C_CiC0[0, 0] / h[2, 0] + (h[0, 0] / h[2, 0]**2) * C_CiC0[2, 0],  # noqa
                -C_CiC0[1, 0] / h[2, 0] + (h[1, 0] / h[2, 0]**2) * C_CiC0[2, 0]   # noqa
            ])
            drdbeta = np.array([
                -C_CiC0[0, 1] / h[2, 0] + (h[0, 0] / h[2, 0]**2) * C_CiC0[2, 1],  # noqa
                -C_CiC0[1, 1] / h[2, 0] + (h[1, 0] / h[2, 0]**2) * C_CiC0[2, 1]   # noqa
            ])
            drdrho = np.array([
                -t_Ci_CiC0[0, 0] / h[2, 0] + (h[0, 0] / h[2, 0]**2) * t_Ci_CiC0[2, 0],  # noqa
                -t_Ci_CiC0[1, 0] / h[2, 0] + (h[1, 0] / h[2, 0]**2) * t_Ci_CiC0[2, 0]   # noqa
            ])
            J[2 * i:(2 * (i + 1)), 0] = drdalpha.ravel()
            J[2 * i:(2 * (i + 1)), 1] = drdbeta.ravel()
            J[2 * i:(2 * (i + 1)), 2] = drdrho.ravel()

        return J

    def reprojection_error(self, x, *args):
        """Reprojection error

        Parameters
        ----------
        cam_model : CameraModel
            Camera model
        track : FeatureTrack
            Feature track
        track_cam_states : list of CameraState
            Camera states where feature track was observed

        Returns
        -------
        r : np.array
            Residual

        """
        cam_model, track, track_cam_states = args

        # Calculate residuals
        N = len(track_cam_states)
        r = zeros((2 * N, 1))
        C_C0G = C(track_cam_states[0].q_CG)
        p_G_C0 = track_cam_states[0].p_G

        alpha, beta, rho = x.ravel()

        for i in range(N):
            # Get camera current rotation and translation
            C_CiG = C(track_cam_states[i].q_CG)
            p_G_Ci = track_cam_states[i].p_G

            # Set camera 0 as origin, work out rotation and translation
            # of camera i relative to to camera 0
            C_CiC0 = dot(C_CiG, C_C0G.T)
            t_Ci_CiC0 = dot(C_CiG, (p_G_C0 - p_G_Ci))

            # Project estimated feature location to image plane
            h = dot(C_CiC0, np.array([[alpha], [beta], [1]])) + rho * t_Ci_CiC0  # noqa

            # Calculate reprojection error
            # -- Convert measurment to image coordinates
            z = cam_model.pixel2image(track.track[i].pt).reshape((2, 1))
            # -- Convert feature location to normalized coordinates
            z_hat = np.array([[h[0, 0] / h[2, 0]], [h[1, 0] / h[2, 0]]])
            # -- Reprojcetion error
            r[2 * i:(2 * (i + 1))] = z - z_hat

        return r.ravel()

    def estimate(self, cam_model, track, track_cam_states, debug=False):
        """Estimate feature 3D location by optimizing over inverse depth
        parameterization using Gauss Newton Optimization

        Parameters
        ----------
        cam_model : CameraModel
            Camera model
        track : FeatureTrack
            Feature track
        track_cam_states : list of CameraState
            Camera states where feature track was observed
        debug :
            Debug mode (default: False)

        Returns
        -------
        p_G_f : np.array - 3x1
            Estimated feature position in global frame

        """
        # # Calculate initial estimate of 3D position
        p_C0_f, residual = self.initial_estimate(cam_model, track,
                                                 track_cam_states)
        print("init: ", p_C0_f.ravel())
        print("residual: ", residual)

        # Get ground truth
        # p_G_f = track.ground_truth

        # Convert ground truth expressed in global frame
        # to be expressed in camera 0
        # C_C0G = C(track_cam_states[0].q_CG)
        # p_G_C0 = track_cam_states[0].p_G
        # p_C0_f = dot(C_C0G, (p_G_f - p_G_C0))

        # print("true: ", p_C0_f.ravel())

        # Create inverse depth params (these are to be optimized)
        alpha = p_C0_f[0, 0] / p_C0_f[2, 0]
        beta = p_C0_f[1, 0] / p_C0_f[2, 0]
        rho = 1.0 / p_C0_f[2, 0]
        theta_k = np.array([alpha, beta, rho])

        # z = 1 / rho
        # X = np.array([[alpha], [beta], [1.0]])
        # C_C0G = C(track_cam_states[0].q_CG)
        # p_G_C0 = track_cam_states[0].p_G
        # init = z * dot(C_C0G.T, X) + p_G_C0

        # Optimize feature location
        args = (cam_model, track, track_cam_states)
        result = least_squares(self.reprojection_error,
                               theta_k,
                               args=args,
                               jac=self.jacobian,
                               verbose=1,
                               method="lm")

        # if result.cost > 1e-4:
        #     return None

        # Calculate feature position in global frame
        alpha, beta, rho = result.x.ravel()
        z = 1 / rho
        X = np.array([[alpha], [beta], [1.0]])
        C_C0G = C(track_cam_states[0].q_CG)
        p_G_C0 = track_cam_states[0].p_G
        p_G_f = z * dot(C_C0G.T, X) + p_G_C0

        # print("ground truth: ", track.ground_truth.ravel())
        # print("cost: ", result.cost)
        # print("initial: ", init.ravel())
        # print("final: ", p_G_f.ravel())

        # p_C_f = dot(C_C0G, (p_G_f - p_G_C0))
        # if p_C_f[2, 0] < 2.0:
        #     return None
        # if p_C_f[2, 0] > 200.0:
        #     return None

        return p_G_f
