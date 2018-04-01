from math import cos
from math import sin
from math import tan

import numpy as np
from numpy import dot

from prototype.utils.utils import rad2deg
from prototype.utils.utils import deg2rad
from prototype.control.quadrotor.waypoint import WaypointController
from prototype.control.quadrotor.position import PositionController
from prototype.control.quadrotor.attitude import AttitudeController


class QuadrotorModel(object):
    """Quadrotor model"""
    def __init__(self, **kwargs):
        self.states = [0.0 for i in range(12)]

        # Moment of inertia in x, y, z axis
        self.Ix = 0.0963
        self.Iy = 0.0963
        self.Iz = 0.1927

        # Rotation and translation drag constant
        self.kr = 0.1
        self.kt = 0.2

        self.arm_length = 0.9  # Quadrotor Arm length
        self.d = 1.0

        self.m = 1.0    # Mass of quadrotor
        self.g = 10.0   # Gravity

        self.ctrl_mode = kwargs.get("ctrl_mode", "POSITION_MODE")

        self.attitude_controller = AttitudeController()
        self.position_controller = PositionController()
        if self.ctrl_mode == "WAYPOINT_MODE":
            waypoints = kwargs["waypoints"]
            look_ahead_dist = kwargs["look_ahead_dist"]
            self.waypoint_controller = WaypointController(waypoints,
                                                          look_ahead_dist)

    @property
    def rpy(self):
        """Roll pitch yaw"""
        return np.array(self.states[0:3])

    @property
    def angular_velocity(self):
        """Angular Velocity"""
        return np.array(self.states[3:6])

    @property
    def position(self):
        """Position"""
        return np.array(self.states[6:9])

    @property
    def velocity(self):
        """Velocity"""
        return np.array(self.states[9:12])

    def update_waypoint_controller(self, dt):
        """ Update waypoint controller

        Parameters
        ----------
        setpoint : np.array
            Position setpoint (x, y, z, yaw)
        dt : float
            Time difference

        Returns
        -------
        motor_inputs : np.array
            Motor inputs (Roll, Pitch, Yaw, Thrust)

        """
        # Waypoint controller
        att_setpoints = self.waypoint_controller.update(self.position,
                                                        self.states[2],
                                                        dt)

        # Attitude controller
        att_actual = np.array([self.states[0],   # Roll
                               self.states[1],   # Pitch
                               self.states[2],   # Yaw
                               self.states[8]])  # z
        motor_inputs = self.attitude_controller.update(
            att_setpoints,
            att_actual,
            dt
        )

        return motor_inputs

    def update_position_controller(self, setpoint, dt):
        """ Update position controller

        Parameters
        ----------
        setpoint : np.array
            Position setpoint (x, y, z, yaw)
        dt : float
            Time difference

        Returns
        -------
        motor_inputs : np.array
            Motor inputs (Roll, Pitch, Yaw, Thrust)

        """
        # Position controller
        pos_actual = np.array([self.states[6],   # x
                               self.states[7],   # y
                               self.states[8],   # z
                               self.states[2]])  # Yaw
        att_setpoints = self.position_controller.update(setpoint,
                                                        pos_actual,
                                                        self.states[2],
                                                        dt)

        # Attitude controller
        att_actual = np.array([self.states[0],   # Roll
                               self.states[1],   # Pitch
                               self.states[2],   # Yaw
                               self.states[8]])  # z
        motor_inputs = self.attitude_controller.update(
            att_setpoints,
            att_actual,
            dt
        )

        return motor_inputs

    def update(self, motor_inputs, dt):
        """Update quadrotor motion model

        Parameters
        ----------
        motor_inputs : np.array
            Motor inputs
        dt : float
            Time difference

        """
        # States
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

        # Convert motor inputs to angular p, q, r and total thrust
        A = np.array([[1.0, 1.0, 1.0, 1.0],
                      [0.0, -self.arm_length, 0.0, self.arm_length],
                      [-self.arm_length, 0.0, self.arm_length, 0.0],
                      [-self.d, self.d, -self.d, self.d]])
        tau = dot(A, motor_inputs)
        tauf = tau[0]
        taup = tau[1]
        tauq = tau[2]
        taur = tau[3]

        # Update
        self.states[0] = ph + (p + q * sin(ph) * tan(th) + r * cos(ph) * tan(th)) * dt  # NOQA
        self.states[1] = th + (q * cos(ph) - r * sin(ph)) * dt  # NOQA
        self.states[2] = ps + ((1 / cos(th)) * (q * sin(ph) + r * cos(ph))) * dt  # NOQA
        self.states[3] = p + (-((self.Iz - self.Iy) / self.Ix) * q * r - (self.kr * p / self.Ix) + (1 / self.Ix) * taup) * dt  # NOQA
        self.states[4] = q + (-((self.Ix - self.Iz) / self.Iy) * p * r - (self.kr * q / self.Iy) + (1 / self.Iy) * tauq) * dt  # NOQA
        self.states[5] = r + (-((self.Iy - self.Ix) / self.Iz) * p * q - (self.kr * r / self.Iz) + (1 / self.Iz) * taur) * dt  # NOQA
        self.states[6] = x + vx * dt
        self.states[7] = y + vy * dt
        self.states[8] = z + vz * dt

        ax = ((-self.kt * vx / self.m) + (1 / self.m) * (cos(ph) * sin(th) * cos(ps) + sin(ph) * sin(ps)) * tauf)  # NOQA
        ay = ((-self.kt * vy / self.m) + (1 / self.m) * (cos(ph) * sin(th) * sin(ps) - sin(ph) * cos(ps)) * tauf)  # NOQA
        az = (-(self.kt * vz / self.m) + (1 / self.m) * (cos(ph) * cos(th)) * tauf - self.g)  # NOQA

        self.states[9] = vx + ax * dt
        self.states[10] = vy + ay * dt
        self.states[11] = vz + az * dt

        # Constrain yaw to be [-180, 180]
        self.states[2] = rad2deg(self.states[2])
        self.states[2] = deg2rad(self.states[2])
