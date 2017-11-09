from math import sqrt
from math import atan2
from math import fmod
from math import pi


def range2d(p1, p2):
    """ Calculate range in 2D

    Args:

        p1 (np.array of size 2): Point 1
        p2 (np.array of size 2): Point 2

    Returns:

        Range between p1 and p2

    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return sqrt(dx**2 + dy**2)


def bearing2d(f, x, theta):
    """ Calculate bearing between feature and robot

    Args:

        f (np.array of size 2): Feature position
        x (np.array of size 2): Robot position
        theta (float): Robot's current heading in radians

    Returns:

        Bearing between feature and robot position

    """
    dx = f[0] - x[0]
    dy = f[1] - x[1]
    # return mod(atan2(dy, dx) - theta + pi, 2 * pi) - pi
    return atan2(dy, dx) - theta


def feature_inview_2d(f, x, theta, rmax, thmax):
    """ Checks to see wheter features is in view of robot

    Args:


        f (np.array of size 2): Feature position
        x (np.array of size 2): Robot position
        theta (float): Robot's current heading in radians
        rmax (float): Max sensor range
        thmax (float): Max sensor bearing

    Returns:

        Boolean to denote whether feature is in view of robot

    """
    # Distance in x and y
    dx = f[0] - x[0]
    dy = f[1] - x[1]

    # Calculate range and bearing
    r = sqrt(dx**2 + dy**2)
    th = fmod(atan2(dy, dx) - theta, 2 * pi)
    if th > pi:
        th = th - 2 * pi

    # Check to see if range and bearing is within view
    if ((r < rmax) and (abs(th) < thmax)):
        return True
    else:
        return False
