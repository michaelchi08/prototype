#!/usr/bin/env python3
import numpy as np


class CarrotController:
    def __init__(self, waypoints, look_ahead_dist):
        if len(waypoints) <= 2:
            raise RuntimeError("Too few waypoints!")

        self.waypoints = waypoints
        self.wp_start = waypoints[0]
        self.wp_end = waypoints[1]
        self.look_ahead_dist = look_ahead_dist

    def closest_point(self, wp_start, wp_end, point):
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
        closest_pt, retval = self.closest_point(wp_start, wp_end, p)

        v = wp_end - wp_start
        u = v / np.linalg.norm(v)
        carrot_pt = closest_pt + r * u

        return carrot_pt, retval

    def update(self, position):
        # calculate new carot point
        carrot_pt, retval = self.carrot_point(
            position,
            self.look_ahead_dist,
            self.wp_start,
            self.wp_end
        )
        print(carrot_pt, retval)

        # update waypoints
        if retval > 1 and len(self.waypoints) > 2:
            self.wp_start = np.array(self.wp_end)
            self.wp_end = np.array(self.waypoints.pop(0))

        return carrot_pt
