from prototype.utils.euler import euler2rot

import numpy as np


class HuskyModel:
    """Husky motion model

    Attributes
    ----------
    p_G : np.array - 3x1
        Position in global frame
    v_G : np.array - 3x1
        Velocity in global frame
    a_G : np.array - 3x1
        Acceleration in global frame
    rpy_G : np.array - 3x1
        Roll, pitch and yaw in global frame
    w_B : np.array - 3x1
        Angular velocity in body frame

    Parameters
    ----------
    pos : np.array - 3x1
        Position in global frame
    vel : np.array - 3x1
        Velocity in global frame
    rpy : np.array - 3x1
        Roll, pitch and yaw in global frame

    """

    def __init__(self, **kwargs):
        # Robot state
        self.p_G = kwargs.get("pos", np.zeros((3, 1)))
        self.v_G = kwargs.get("vel", np.zeros((3, 1)))
        self.a_G = np.zeros((3, 1))
        self.rpy_G = kwargs.get("rpy", np.zeros((3, 1)))

        # IMU measurement
        self.v_B = np.zeros((3, 1))
        self.a_B = np.zeros((3, 1))
        self.w_B = np.zeros((3, 1))

    def update(self, v_B, w_B, dt):
        """Update motion model

        Parameters
        ----------
        v_B : np.array - 3x1
            Velocity in body frame
        w_B : np.array - 3x1
            Angular velocity in body frame

        """
        # Update robot state
        # -- Position
        p_kp1_G = self.p_G + self.v_G * dt
        # -- Velocity
        R_BG = euler2rot(self.rpy_G, 321)
        v_kp1_G = np.dot(R_BG, v_B)
        # -- Acceleration
        a_kp1_G = v_kp1_G - self.v_G
        # -- Orientation
        rpy_kp1_G = self.rpy_G + w_B * dt
        # -- Finish up
        self.p_G = p_kp1_G
        self.v_G = v_kp1_G
        self.a_G = a_kp1_G
        self.rpy_G = rpy_kp1_G

        # Update IMU measurements
        # -- Acceleration
        self.a_B = np.dot(R_BG.T, self.a_G)
        # -- Velocity
        self.v_B = v_B
        # -- Angular velocity
        self.w_B = w_B
