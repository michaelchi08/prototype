import numpy as np

from prototype.control.pid import PID
from prototype.utils.utils import rad2deg
from prototype.utils.utils import deg2rad


class AttitudeController(object):
    """Attitude Controller

    Attributes
    ----------
    roll_controller : PID
        PID controller for roll
    pitch_controller : PID
        PID controller for pitch
    yaw_controller : PID
        PID controller for yaw

    dt : float
        Time difference
    outputs : np.array - 1x4
        Position controller output where the elements represent roll, pitch, yaw
        and throttle

    """

    def __init__(self):
        self.roll_controller = PID(200.0, 0.5, 10.0)
        self.pitch_controller = PID(200.0, 0.5, 10.0)
        self.yaw_controller = PID(200.0, 0.5, 10.0)

        self.dt = 0.0
        self.outputs = np.array([0.0, 0.0, 0.0, 0.0])

    def update(self, setpoints, actual, dt):
        """Update

        Parameters
        ----------
        setpoints : np.array - 1x3
            Attitude setpoints in body frame (roll, pitch, yaw, z)
        actual : np.array - 1x3
            Actual attitude in body frame (roll, pitch, yaw, z)
        yaw : float
            Yaw setpoint
        dt : float
            Time difference

        Returns
        -------
        outputs : np.array - 1x4
            Attitude controller output where the elements represent roll, pitch,
            yaw and throttle

        """
        self.dt += dt
        if self.dt < 0.001:
            return self.outputs

        # update yaw error
        actual_yaw = rad2deg(actual[2])
        setpoint_yaw = rad2deg(setpoints[2])
        error_yaw = setpoint_yaw - actual_yaw
        if error_yaw > 180.0:
            error_yaw -= 360.0
        elif error_yaw < -180.0:
            error_yaw += 360.0
        error_yaw = deg2rad(error_yaw)

        # roll pitch yaw
        r = self.roll_controller.update(setpoints[0], actual[0], self.dt)
        p = self.pitch_controller.update(setpoints[1], actual[1], self.dt)
        y = self.yaw_controller.update(error_yaw, 0.0, self.dt)

        # thrust
        max_thrust = 5.0
        t = max_thrust * setpoints[3]  # convert relative to true thrust
        t = max_thrust if t > max_thrust else t  # limit thrust
        t = 0.0 if t < 0 else t                  # limit thrust

        # map roll, pitch, yaw and thrust to motor outputs
        outputs = [0.0, 0.0, 0.0, 0.0]
        outputs[0] = -p - y + t
        outputs[1] = -r + y + t
        outputs[2] = p - y + t
        outputs[3] = r + y + t

        # limit outputs
        for i in range(4):
            if outputs[i] > max_thrust:
                outputs[i] = max_thrust
            elif outputs[i] < 0.0:
                outputs[i] = 0.0

        # keep track of outputs
        self.outputs = outputs
        self.dt = 0.0

        return outputs
