import numpy as np


class CarrotController(object):
    """Carrot controller"""
    def __init__(self, waypoints, look_ahead_dist):
        if waypoints.shape[0] <= 2:
            raise RuntimeError("Too few waypoints!")

        self.waypoints = waypoints
        self.wp_start = waypoints[0]
        self.wp_end = waypoints[1]
        self.wp_index = 1
        self.look_ahead_dist = look_ahead_dist

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

    def carrot_point(self, p_G, r, wp_start, wp_end):
        """Calculate carrot point

        Parameters
        ----------
        p_G : numpy array
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
        closest_pt, retval = self.closest_point(wp_start, wp_end, p_G)

        v = wp_end - wp_start
        u = v / np.linalg.norm(v)
        carrot_pt = closest_pt + r * u

        return carrot_pt, retval

    def update(self, p_G):
        """Update carrot controller

        Parameters
        ----------
        p_G : numpy array
            robot position

        Returns
        -------
        carrot_pt : numpy array
            carrot point

        """
        # Calculate new carot point
        carrot_pt, retval = self.carrot_point(
            p_G,
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
