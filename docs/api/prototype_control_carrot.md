# prototype.control.carrot

## Classes

- CarrotController


### CarrotController
Carrot controller 

---

    __init__(self, waypoints, look_ahead_dist)


---

    carrot_point(self, p, r, wp_start, wp_end)

Calculate carrot point

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

        

---

    closest_point(self, wp_start, wp_end, point)

Calculate closest point between waypoint

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

        

---

    update(self, position)

Update carrot controller

        Args:

            position (numpy array): robot position

        Returns:

            carrot_pt (numpy array): carrot point

        

---

