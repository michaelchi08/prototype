from math import fabs

import numpy as np
from numpy import dot

from prototype.control.pid import PID
from prototype.utils.utils import deg2rad
from prototype.utils.euler import euler2rot


class PositionController(object):
    """Position Controller

    Attributes
    ----------
    x_controller : PID
        PID controller for x-axis
    y_controller : PID
        PID controller for y-axis
    z_controller : PID
        PID controller for z-axis

    dt : float
        Time difference
    outputs : np.array - 1x4
        Position controller output where the elements represent roll, pitch, yaw
        and throttle

    """

    def __init__(self):
        self.x_controller = PID(0.5, 0.0, 0.035)
        self.y_controller = PID(0.5, 0.0, 0.035)
        self.z_controller = PID(0.3, 0.0, 0.018)

        self.dt = 0.0
        self.outputs = np.array([0.0, 0.0, 0.0, 0.0])

    def update(self, setpoints, actual, yaw, dt):
        """Update controller

        Parameters
        ----------
        setpoints : np.array - 1x3
            Position setpoints in world frame (x, y, z)
        actual : np.array - 1x3
            Actual position in world frame (x, y, z)
        yaw : float
            Yaw setpoint
        dt : float
            Time difference

        Returns
        -------
        outputs : np.array - 1x4
            Position controller output where the elements represent roll, pitch,
            yaw and throttle

        """
        # Check rate
        self.dt += dt
        if self.dt < 0.01:
            return self.outputs

        # Update RPY errors relative to quadrotor by incorporating yaw
        errors = [0.0, 0.0, 0.0]
        errors[0] = setpoints[0] - actual[0]
        errors[1] = setpoints[1] - actual[1]
        errors[2] = setpoints[2] - actual[2]
        euler = [0.0, 0.0, actual[3]]
        R = euler2rot(euler, 123)
        errors = dot(R, errors)

        # Roll, pitch, yaw and thrust
        r = -1.0 * self.y_controller.update(errors[1], 0.0, dt)
        p = self.x_controller.update(errors[0], 0.0, dt)
        y = yaw
        t = 0.5 + self.z_controller.update(errors[2], 0.0, dt)
        outputs = [r, p, y, t]

        # Limit roll, pitch
        for i in range(2):
            if outputs[i] > deg2rad(30.0):
                outputs[i] = deg2rad(30.0)
            elif outputs[i] < deg2rad(-30.0):
                outputs[i] = deg2rad(-30.0)

        # Limit thrust
        if outputs[3] > 1.0:
            outputs[3] = 1.0
        elif outputs[3] < 0.0:
            outputs[3] = 0.0

        # Yaw first if threshold reached
        if fabs(yaw - actual[3]) > deg2rad(2.0):
            outputs[0] = 0.0
            outputs[1] = 0.0

        # Keep track of outputs
        self.outputs = outputs
        self.dt = 0.0

        return outputs
