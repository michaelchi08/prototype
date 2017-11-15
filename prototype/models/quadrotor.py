from math import cos
from math import sin
from math import tan

import numpy as np
from numpy import dot

from prototype.utils.utils import rad2deg
from prototype.utils.utils import deg2rad
from prototype.control.quadrotor.position import PositionController
from prototype.control.quadrotor.attitude import AttitudeController


class QuadrotorModel(object):
    """Quadrotor model"""

    def __init__(self):
        self.states = [0.0 for i in range(12)]

        self.Ix = 0.0963
        self.Iy = 0.0963
        self.Iz = 0.1927

        self.kr = 0.1
        self.kt = 0.2

        self.l = 0.9
        self.d = 1.0

        self.m = 1.0
        self.g = 10.0

        self.position_controller = PositionController()
        self.attitude_controller = AttitudeController()

    @property
    def attitude(self):
        """Attitude"""
        roll = self.states[0]
        pitch = self.states[1]
        yaw = self.states[2]
        att = np.array([[roll], [pitch], [yaw]])
        return att

    @property
    def angular_velocity(self):
        """Angular Velocity"""
        wx = self.states[3]
        wy = self.states[4]
        wz = self.states[5]
        w = np.array([[wx], [wy], [wz]])
        return w

    @property
    def position(self):
        """Position"""
        x = self.states[6]
        y = self.states[7]
        z = self.states[8]
        pos = np.array([[x], [y], [z]])
        return pos

    @property
    def velocity(self):
        """Velocity"""
        vx = self.states[9]
        vy = self.states[10]
        vz = self.states[11]
        vel = np.array([[vx], [vy], [vz]])
        return vel

    def update(self, motor_inputs, dt):
        """Update quadrotor motion model

        Parameters
        ----------
        motor_inputs : np.array
            Motor inputs
        dt : float
            Time difference

        """
        # states
        ph = self.states[0]
        th = self.states[1]
        ps = self.states[2]

        p = self.states[3]
        q = self.states[4]
        r = self.states[5]

        x = self.states[6]  # NOQA
        y = self.states[7]  # NOQA
        z = self.states[8]  # NOQA

        vx = self.states[9]
        vy = self.states[10]
        vz = self.states[11]

        # convert motor inputs to angular p, q, r and total thrust
        A = np.array([[1.0, 1.0, 1.0, 1.0],
                      [0.0, -self.l, 0.0, self.l],
                      [-self.l, 0.0, self.l, 0.0],
                      [-self.d, self.d, -self.d, self.d]])
        tau = dot(A, motor_inputs)
        tauf = tau[0]
        taup = tau[1]
        tauq = tau[2]
        taur = tau[3]

        # update
        self.states[0] = ph + (p + q * sin(ph) * tan(th) + r * cos(ph) * tan(th)) * dt  # NOQA
        self.states[1] = th + (q * cos(ph) - r * sin(ph)) * dt  # NOQA
        self.states[2] = ps + ((1 / cos(th)) * (q * sin(ph) + r * cos(ph))) * dt  # NOQA
        self.states[3] = p + (-((self.Iz - self.Iy) / self.Ix) * q * r - (self.kr * p / self.Ix) + (1 / self.Ix) * taup) * dt  # NOQA
        self.states[4] = q + (-((self.Ix - self.Iz) / self.Iy) * p * r - (self.kr * q / self.Iy) + (1 / self.Iy) * tauq) * dt  # NOQA
        self.states[5] = r + (-((self.Iy - self.Ix) / self.Iz) * p * q - (self.kr * r / self.Iz) + (1 / self.Iz) * taur) * dt  # NOQA
        self.states[6] = x + vx * dt
        self.states[7] = y + vy * dt
        self.states[8] = z + vz * dt

        self.states[9] = vx + ((-self.kt * vx / self.m) + (1 / self.m) * (cos(ph) * sin(th) * cos(ps) + sin(ph) * sin(ps)) * tauf) * dt  # NOQA
        self.states[10] = vy + ((-self.kt * vy / self.m) + (1 / self.m) * (cos(ph) * sin(th) * sin(ps) - sin(ph) * cos(ps)) * tauf) * dt  # NOQA
        self.states[11] = vz + (-(self.kt * vz / self.m) + (1 / self.m) * (cos(ph) * cos(th)) * tauf - self.g) * dt  # NOQA

        # constrain yaw to be [-180, 180]
        self.states[2] = rad2deg(self.states[2])
        self.states[2] = deg2rad(self.states[2])
