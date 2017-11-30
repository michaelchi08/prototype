from math import pi


def circle_trajectory(r, v):
    """Calculate target angular velocity given a desired circle
    trajectory of radius r and velocity v

    Parameters
    ----------
    r : float
        Desired circle radius
    v : float
        Desired trajectory velocity

    Returns
    -------
    w : float
        Target angular velocity to complete a circle of radius r and
        velocity v

    """
    dist = 2 * pi * r
    time = dist / v
    w = (2 * pi) / time
    return w
