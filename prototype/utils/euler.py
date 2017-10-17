def rotx(theta):
    """ Rotation matrix around x-axis (counter-clockwise)

    Args:

        theta (float): Rotation around x in radians

    Returns:

        3 x 3 Rotation matrix

    """
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, cos(theta), -sin(theta)],
                     [0.0, sin(theta), cos(theta)]])


def roty(theta):
    """ Rotation matrix around y-axis (counter-clockwise)

    Args:

        theta (float): Rotation around y in radians

    Returns:

        3 x 3 Rotation matrix

    """
    return np.array([[cos(theta), 0.0, sin(theta)],
                     [0.0, 1.0, 0.0],
                     [-sin(theta), 0.0, cos(theta)]])


def rotz(theta):
    """ Rotation matrix around z-axis (counter-clockwise)

    Args:

        theta (float): Rotation around y in radians

    Returns:

        3 x 3 Rotation matrix

    """
    return np.array([[cos(theta), -sin(theta), 0.0],
                     [sin(theta), cos(theta), 0.0],
                     [0.0, 0.0, 1.0]])


def euler2rot(euler, euler_seq):
    """ Convert euler to rotation matrix R
    This function assumes we are performing an extrinsic rotation.

    Source:
        Euler Angles, Quaternions and Transformation Matrices
        https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf
    """
    if euler_seq == 123:
        t1, t2, t3 = euler

        R11 = cos(t2) * cos(t3)
        R12 = -cos(t2) * sin(t3)
        R13 = sin(t2)

        R21 = sin(t1) * sin(t2) * cos(t3) + cos(t1) * sin(t3)
        R22 = -sin(t1) * sin(t2) * sin(t3) + cos(t1) * cos(t3)
        R23 = -sin(t1) * cos(t2)

        R31 = -cos(t1) * sin(t2) * cos(t3) + sin(t1) * sin(t3)
        R32 = cos(t1) * sin(t2) * sin(t3) + sin(t1) * cos(t3)
        R33 = cos(t1) * cos(t2)

    elif euler_seq == 321:
        t3, t2, t1 = euler

        R11 = cos(t1) * cos(t2)
        R12 = cos(t1) * sin(t2) * sin(t3) - sin(t1) * cos(t3)
        R13 = cos(t1) * sin(t2) * cos(t3) + sin(t1) * sin(t3)

        R21 = sin(t1) * cos(t2)
        R22 = sin(t1) * sin(t2) * sin(t3) + cos(t1) * cos(t3)
        R23 = sin(t1) * sin(t2) * cos(t3) - cos(t1) * sin(t3)

        R31 = -sin(t2)
        R32 = cos(t2) * sin(t3)
        R33 = cos(t2) * cos(t3)

    else:
        error_msg = "Error! Unsupported euler sequence [%s]" % str(euler_seq)
        raise RuntimeError(error_msg)

    return np.array([[R11, R12, R13],
                     [R21, R22, R23],
                     [R31, R32, R33]])


def euler2quat(euler, euler_seq):
    """ Convert euler to quaternion
    This function assumes we are performing an extrinsic rotation.

    Source:
        Euler Angles, Quaternions and Transformation Matrices
        https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf
    """

    if euler_seq == 123:
        t1, t2, t3 = euler
        c1 = cos(t1 / 2.0)
        c2 = cos(t2 / 2.0)
        c3 = cos(t3 / 2.0)
        s1 = sin(t1 / 2.0)
        s2 = sin(t2 / 2.0)
        s3 = sin(t3 / 2.0)

        w = -s1 * s2 * s3 + c1 * c2 * c3
        x = s1 * c2 * c3 + s2 * s3 * c1
        y = -s1 * s3 * c2 + s2 * c1 * c3
        z = s1 * s2 * c3 + s3 * c1 * c2
        return quatnormalize([w, x, y, z])

    elif euler_seq == 321:
        t3, t2, t1 = euler
        c1 = cos(t1 / 2.0)
        c2 = cos(t2 / 2.0)
        c3 = cos(t3 / 2.0)
        s1 = sin(t1 / 2.0)
        s2 = sin(t2 / 2.0)
        s3 = sin(t3 / 2.0)

        w = s1 * s2 * s3 + c1 * c2 * c3
        x = -s1 * s2 * c3 + s3 * c1 * c2
        y = s1 * s3 * c2 + s2 * c1 * c3
        z = s1 * c2 * c3 - s1 * s3 * c1
        return quatnormalize([w, x, y, z])

    else:
        error_msg = "Error! Unsupported euler sequence [%s]" % str(euler_seq)
        raise RuntimeError(error_msg)
