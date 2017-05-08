#!/usr/bin/env python3
from math import pi
from math import atan2

import numpy as np


def atan3(y, x):
    angle = atan2(y, x)

    if y < 0:
        angle = angle + 2 * pi

    return angle


# def calculate_delta(x_t, carrot_t, delta_max):
#     theta = x_t(3)
#
#     # calculate angle between carrot and bicycle
#     x = (carrot_t(1) - x_t(1))
#     y = (carrot_t(2) - x_t(2))
#     angle_of_vec = atan3(y, x)  # returns only +ve angle
#
#     # limit delta_t to pi and -pi only
#     delta_t = -(theta - angle_of_vec)
#     delta_t = mod(delta_t + pi, 2 * pi) - pi
#
#     # limit delta_t to steering angle max
#     if delta_t > delta_max:
#         delta_t = delta_max
#     elif delta_t < -delta_max:
#         delta_t = -delta_max
#
#     return theta


# def waypoint_reached(position, waypoint, threshold):
#     if dist_between_points(position, waypoint) < threshold:
#         return True
#     else:
#         return False


class CarrotController(object):
    """ Carrot controller """

    def __init__(self, waypoints, look_ahead_dist):
        if len(waypoints) <= 2:
            raise RuntimeError("Too few waypoints!")

        self.waypoints = waypoints
        self.wp_start = waypoints[0]
        self.wp_end = waypoints[1]
        self.look_ahead_dist = look_ahead_dist

    def closest_point(self, wp_start, wp_end, point):
        """ Calculate closest point between waypoint

        Args:

            wp_start (numpy array): waypoint start
            wp_end (numpy array): waypoint end
            point (numpy array): robot position

        Returns:

            (closest_point, retval)

        where `closest_point` is a 2D vector of the closest point and `retval`
        denotes whether `closest_point` is

            1. before wp_start
            2. after wp_end
            3. middle of wp_start and wp_end

        """
        # calculate closest point
        v1 = point - wp_start
        v2 = wp_end - wp_start
        t = v1.dot(v2) / pow(np.linalg.norm(v2), 2)
        closest = wp_start + t * v2

        # check if point p is:
        # 1. before wp_start
        # 2. after wp_end
        # 3. middle of wp_start and wp_end
        if t < 0.0:
            return (closest, 1)
        elif t > 1.0:
            return (closest, 2)
        else:
            return (closest, 0)

    def carrot_point(self, p, r, wp_start, wp_end):
        """ Calculate carrot point

        Args:

            p (numpy array): robot pose
            r (numpy array): look ahead distance
            wp_start (numpy array): waypoint start
            wp_end (numpy array): waypoint end

        Returns:

            (carrot_pt, retval)

        where `carrot_pt` is a 2D vector of the carrot point and `retval`
        denotes whether `carrot_pt` is

            1. before wp_start
            2. after wp_end
            3. middle of wp_start and wp_end

        """
        closest_pt, retval = self.closest_point(wp_start, wp_end, p)

        v = wp_end - wp_start
        u = v / np.linalg.norm(v)
        carrot_pt = closest_pt + r * u

        return carrot_pt, retval

    def update(self, position):
        """ Update carrot controller

        Args:

            position (numpy array): robot position

        Returns:

            carrot_pt (numpy array): carrot point

        """
        # calculate new carot point
        carrot_pt, retval = self.carrot_point(
            position,
            self.look_ahead_dist,
            self.wp_start,
            self.wp_end
        )

        # update waypoints
        if retval > 1 and len(self.waypoints) > 2:
            self.wp_start = np.array(self.wp_end)
            self.wp_end = np.array(self.waypoints.pop(0))

        return carrot_pt
