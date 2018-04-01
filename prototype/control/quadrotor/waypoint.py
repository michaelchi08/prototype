from math import pi
from math import atan2
from math import fabs

import numpy as np

from prototype.control.pid import PID
from prototype.utils.utils import deg2rad
from prototype.utils.euler import euler2rot


class WaypointController(object):
    """Waypoint Controller

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
        Position controller output where the elements represent roll, pitch,
        yaw and throttle

    """

    def __init__(self, waypoints, look_ahead_dist):
        if waypoints.shape[0] <= 2:
            raise RuntimeError("Too few waypoints!")

        self.waypoints = waypoints
        self.wp_start = waypoints[0]
        self.wp_end = waypoints[1]
        self.wp_index = 1
        self.look_ahead_dist = look_ahead_dist

        self.at_ctrl = PID(0.3, 0.0, 0.05)
        self.ct_ctrl = PID(0.3, 0.0, 0.05)
        self.z_ctrl = PID(0.3, 0.0, 0.035)
        self.yaw_ctrl = PID(0.1, 0.0, 0.0)

        self.dt = 0.0
        self.outputs = np.array([0.0, 0.0, 0.0, 0.0])

    def closest_point(self, wp_start, wp_end, point):
        """Calculate closest point between waypoint

        Parameters
        ----------
        wp_start : numpy array
            waypoint start
        wp_end : numpy array
            waypoint end
        point : numpy array
            robot position

        Returns
        -------
        (closest_point, retval) : (np.array, int)
            where `closest_point` is a 2D vector of the closest point and
            `retval` denotes whether `closest_point` is:
            - 1. before wp_start
            - 2. after wp_end
            - 3. middle of wp_start and wp_end

        """
        # Calculate closest point
        v1 = point - wp_start
        v2 = wp_end - wp_start
        t = v1.dot(v2) / pow(np.linalg.norm(v2), 2)
        closest_pt = wp_start + t * v2

        # Check if point p is:
        # 1. before wp_start
        # 2. after wp_end
        # 3. middle of wp_start and wp_end
        if t < 0.0:
            return (closest_pt, 1)
        elif t > 1.0:
            return (closest_pt, 2)
        else:
            return (closest_pt, 0)

    def carrot_point(self, pos, r, wp_start, wp_end):
        """Calculate carrot point

        Parameters
        ----------
        pos : numpy array
            robot pose
        r : numpy array
            look ahead distance
        wp_start : numpy array
            waypoint start
        wp_end : numpy array
            waypoint end

        Returns
        -------
        (carrot_pt, retval) : (np.array, int)
            where `carrot_pt` is a 2D vector of the carrot point and `retval`
            denotes whether `carrot_pt` is
            1. before wp_start
            2. after wp_end
            3. middle of wp_start and wp_end

        """
        closest_pt, retval = self.closest_point(wp_start, wp_end, pos)

        v = wp_end - wp_start
        u = v / np.linalg.norm(v)
        carrot_pt = closest_pt + r * u

        return carrot_pt, retval

    def update_waypoint(self, pos):
        """Update waypoint

        Parameters
        ----------
        pos : np.array
            Position

        Returns
        -------
        carrot_pt : np.array
            Waypoint

        """
        # Calculate new carot point
        carrot_pt, retval = self.carrot_point(
            pos,
            self.look_ahead_dist,
            self.wp_start,
            self.wp_end
        )

        # Update waypoints
        if retval > 1 and (self.waypoints.shape[0] - 1) != self.wp_index:
            self.wp_index += 1
            self.wp_start = np.array(self.wp_end)
            self.wp_end = np.array(self.waypoints[self.wp_index])
        elif (self.waypoints.shape[0] - 1) == self.wp_index:
            carrot_pt = self.waypoints[-1]

        return carrot_pt

    def calc_yaw_to_waypoint(self, wp, pos):
        """Calculate yaw to waypoint

        Parameters
        ----------
        wp : np.array
            Waypoint
        pos : np.array
            Current position

        """
        dx = wp[0] - pos[0]
        dy = wp[1] - pos[1]

        heading = atan2(dy, dx)
        if heading > pi:
            heading -= 2 * pi
        elif heading < -pi:
            heading += 2 * pi

        return heading

    def update(self, pos, yaw, dt):
        """Update waypoint controller

        Parameters
        ----------
        pos : numpy array
            robot position
        yaw : float
            Yaw setpoint
        dt : float
            Time difference

        Returns
        -------
        carrot_pt : numpy array
            carrot point

        """
        # Time pre-check
        self.dt += dt
        if self.dt < 0.01:
            return self.outputs

        # Get new waypoint
        wp = self.update_waypoint(pos)

        # Calculate yaw error
        # actual_yaw = yaw
        # setpoint_yaw = self.calc_yaw_to_waypoint(wp, pos)
        # error_yaw = setpoint_yaw - actual_yaw
        # if error_yaw > pi:
        #     error_yaw -= 2 * pi
        # elif error_yaw < -pi:
        #     error_yaw += 2 * pi

        # Calculate along and cross track errors
        # R_BG = euler2rot(np.array([0.0, 0.0, yaw]), 321).T
        # error_p_B = np.dot(R_BG, (wp - pos))
        # error_at = error_p_B[1]
        # error_ct = error_p_B[0]
        # error_z = error_p_B[2]
        error_at = wp[0] - pos[0]
        error_ct = wp[1] - pos[1]
        error_z = wp[2] - pos[2]

        # Roll, pitch, yaw and thrust
        r = -1 * self.ct_ctrl.update(error_ct, 0.0, self.dt)
        p = self.at_ctrl.update(error_at, 0.0, self.dt)
        # r = 0.0
        # p = 0.0
        # y = self.yaw_ctrl.update(error_yaw, 0.0, self.dt)
        y = 0.0
        t = 0.5 + self.z_ctrl.update(error_z, 0.0, self.dt)
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

        # # Yaw first if threshold reached
        # if fabs(yaw - actual[3]) > deg2rad(2.0):
        #     outputs[0] = 0.0
        #     outputs[1] = 0.0

        # Keep track of outputs
        self.outputs = outputs
        self.dt = 0.0

        return outputs
