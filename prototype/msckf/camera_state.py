from math import sqrt

import numpy as np
from numpy import dot

from prototype.utils.quaternion.jpl import quatlcomp
from prototype.utils.quaternion.jpl import quatnormalize


class CameraState:
    """Camera state

    Parameters
    ----------
    p_G : np.array
        Position of camera in Global frame
    q_CG : np.array
        Orientation of camera in Global frame

    Attributes
    ----------
    p_G : np.array
        Position of camera in Global frame
    q_CG : np.array
        Orientation of camera in Global frame
    tracks : :obj`list` of :obj`FeatureTrack`
        Feature tracks

    """

    def __init__(self, frame_id, q_CG, p_G):
        self.size = 6  # Number of elements in state vector
        self.frame_id = frame_id

        # State vector
        self.q_CG = np.array(q_CG).reshape((4, 1))
        self.p_G = np.array(p_G).reshape((3, 1))

    def correct(self, dx):
        """Correct the camera state

        Parameters
        ----------
        dx : np.array - 6x1
            Camera state correction, where
            dtheta_IG = dx[0:3]
            dp_G = dx[3:6]

        """
        # Split dx into its own components
        dtheta_CG = dx[0:3]
        dp_G = dx[3:6]

        # Time derivative of quaternion (small angle approx)
        dq_CG = 0.5 * dtheta_CG
        norm = dot(dq_CG.T, dq_CG)
        if norm > 1.0:
            dq_CG = np.block([[dq_CG], [1.0]]) / sqrt(1.0 + norm)
        else:
            dq_CG = np.block([[dq_CG], [sqrt(1.0 - norm)]])
        dq_CG = quatnormalize(dq_CG)

        # Correct camera state
        self.q_CG = dot(quatlcomp(dq_CG), self.q_CG)
        self.p_G = self.p_G + dp_G

    def __str__(self):
        s = "Camera state:\n"
        s += "frame_id:\t{}\n".format(str(self.frame_id))
        s += "q:\t{}\n".format(str(np.round(self.q_CG, 4).ravel()))
        s += "p:\t{}\n".format(str(np.round(self.p_G, 4).ravel()))
        return s
