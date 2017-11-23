import numpy as np
from numpy import zeros
from numpy import dot
from numpy.linalg import inv

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
    def __init__(self, **kwargs):
        self.max_iter = kwargs.get("max_iter", 30)

    def triangulate(self, pt1, pt2, C_C0C1, t_C0_C1C0):
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
        pt1 = pt1 / np.linalg.norm(pt1)
        pt2 = pt2 / np.linalg.norm(pt2)

        # Triangulate
        A = np.block([pt1, -dot(C_C0C1, pt2)])
        b = t_C0_C1C0
        result = np.linalg.lstsq(A, b)
        x, residual, rank, s = result
        p_C0_f = x[0] * pt1

        return p_C0_f

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
        k : int
            Optimized over k iterations
        r : np.array - 2Nx1
            Residual over camera states where feature was tracked, where N is
            the length of the feature track

        """
        # Calculate rotation and translation of first and last camera states
        C_C0G = C(track_cam_states[0].q_CG)
        C_C1G = C(track_cam_states[-1].q_CG)
        p_G_C0 = track_cam_states[0].p_G
        p_G_C1 = track_cam_states[-1].p_G
        # -- Obtain rotation and translation from camera 0 to camera 1
        C_C0C1 = dot(C_C0G, C_C1G.T)
        t_C1_C1C0 = dot(C_C0G, (p_G_C1 - p_G_C0))
        # -- Convert from pixel coordinates to image coordinates
        pt1 = cam_model.pixel2image(track.track[0].pt).reshape((2, 1))
        pt2 = cam_model.pixel2image(track.track[-1].pt).reshape((2, 1))

        # Calculate initial estimate of 3D position
        p_C0_f = self.triangulate(pt1, pt2, C_C0C1, t_C1_C1C0)
        print("p_C0_f: ", p_C0_f)

        # Create inverse depth params (these are to be optimized)
        alpha = p_C0_f[0, 0] / p_C0_f[2, 0]
        beta = p_C0_f[1, 0] / p_C0_f[2, 0]
        rho = 1.0 / p_C0_f[2, 0]

        # Gauss Newton optimization
        r_Jprev = float("inf")  # residual jacobian

        for k in range(self.max_iter):
            N = len(track_cam_states)
            r = zeros((2 * N, 1))
            J = zeros((2 * N, 3))
            W = zeros((2 * N, 2 * N))

            # Calculate residuals
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
                x = np.array([h[0] / h[2], h[1] / h[2]])
                # -- Reprojcetion error
                r[2 * i:(2 * (i + 1))] = z - x

                # Form the Jacobian
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

                # Form the weight matrix
                W[2 * i:(2 * (i + 1)), 2 * i:(2 * (i + 1))] = np.diag([0.00001, 0.00001]) # noqa

            # Check hessian if it is numerically ok
            H_approx = dot(J.T, J)
            if np.linalg.cond(H_approx) > 1e4:
                return None

            # Update estimate params using Gauss Newton
            delta = dot(inv(H_approx), dot(J.T, r))
            theta_k = np.array([[alpha], [beta], [rho]]) - delta
            alpha = theta_k[0, 0]
            beta = theta_k[1, 0]
            rho = theta_k[2, 0]

            # Check how fast the residuals are converging to 0
            r_Jnew = float(0.5 * dot(r.T, r))
            if r_Jnew < 0.0001:
                break
            r_J = abs((r_Jnew - r_Jprev) / r_Jnew)
            r_Jprev = r_Jnew

            # Break loop if not making any progress
            if r_J < 0.0001:
                break

            # # Update params using Weighted Gauss Newton
            # JWJ = dot(J.T, np.linalg.solve(W, J))
            # delta = np.linalg.solve(JWJ, dot(-J.T, np.linalg.solve(W, r)))
            # theta_k = np.array([[alpha], [beta], [rho]]) + delta
            # alpha = theta_k[0, 0]
            # beta = theta_k[1, 0]
            # rho = theta_k[2, 0]
            #
            # # Check how fast the residuals are converging to 0
            # r_Jnew = float(0.5 * dot(r.T, r))
            # if r_Jnew < 0.0000001:
            #     break
            # r_J = abs((r_Jnew - r_Jprev) / r_Jnew)
            # r_Jprev = r_Jnew
            #
            # # Break loop if not making any progress
            # if r_J < 0.000001:
            #     break

        # Debug
        if debug:
            # print(k)
            # print(alpha)
            # print(beta)
            # print(rho)
            print("track_length: ", track.tracked_length())
            print("iterations: ", k)
            print("residual: ", r)

        # Convert estimated inverse depth params back to feature position in
        # global frame.  See (Eq.38, Mourikis2007 (A Multi-State Constraint
        # Kalman Filter for Vision-aided Inertial Navigation)
        z = 1 / rho
        X = np.array([[alpha], [beta], [1.0]])
        p_G_f = z * dot(C_C0G.T, X) + p_G_C0

        return p_G_f
