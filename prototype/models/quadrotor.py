#!/usr/bin/env python3
from math import cos
from math import sin
from math import tan

import numpy as np

from prototype.utils.maths import rad2deg
from prototype.utils.maths import deg2rad


class QuadrotorModel(object):
    def __init__(self):
        self.states = None

        self.Ix = 0.0
        self.Iy = 0.0
        self.Iz = 0.0

        self.kr = 0.0
        self.kt = 0.0

        self.m = 0.0
        self.g = 0.0

    def update(self, motor_inputs, dt):
        # states
        ph = self.states[0]
        th = self.states[1]
        ps = self.states[2]

        p = self.states[3]
        q = self.states[4]
        r = self.states[5]

        x = self.states[6]  # noqa
        y = self.states[7]  # noqa
        z = self.states[8]  # noqa

        vx = self.states[9]
        vy = self.states[10]
        vz = self.states[11]

        # convert motor inputs to angular p, q, r and total thrust
        A = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [0.0, -self.l, 0.0, self.l],
            [-self.l, 0.0, self.l, 0.0],
            [-self.d, self.d, -self.d, self.d]
        ])

        tau = A * motor_inputs
        tauf = tau[0]
        taup = tau[1]
        tauq = tau[2]
        taur = tau[3]

        # update
        self.states[0] = ph + (p + q * sin(ph) * tan(th) + r * cos(ph) * tan(th)) * dt  # noqa
        self.states[1] = th + (q * cos(ph) - r * sin(ph)) * dt  # noqa
        self.states[2] = ps + ((1 / cos(th)) * (q * sin(ph) + r * cos(ph))) * dt  # noqa

        self.states[3] = p + (-((Iz - Iy) / Ix) * q * r - (kr * p / Ix) + (1 / Ix) * taup) * dt  # noqa
        self.states[4] = q + (-((Ix - Iz) / Iy) * p * r - (kr * q / Iy) + (1 / Iy) * tauq) * dt  # noqa
        self.states[5] = r + (-((Iy - Ix) / Iz) * p * q - (kr * r / Iz) + (1 / Iz) * taur) * dt  # noqa

        self.states[6] = x + vx * dt
        self.states[7] = y + vy * dt
        self.states[8] = z + vz * dt

        self.states[9] = vx + ((-kt * vx / m) + (1 / m) * (cos(ph) * sin(th) * cos(ps) + sin(ph) * sin(ps)) * tauf) * dt  # noqa
        self.states[10] = vy + ((-kt * vy / m) + (1 / m) * (cos(ph) * sin(th) * sin(ps) - sin(ph) * cos(ps)) * tauf) * dt  # noqa
        self.states[11] = vz + (-(kt * vz / m) + (1 / m) * (cos(ph) * cos(th)) * tauf - g) * dt  # noqa

        # constrain yaw to be [-180, 180]
        self.states[2] = rad2deg(self.states[2])
        self.states[2] = deg2rad(self.states[2])
