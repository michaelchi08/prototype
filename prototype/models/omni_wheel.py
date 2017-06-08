from math import cos
from math import sin
from math import pi

import numpy as np
from numpy import dot
from numpy.linalg import inv


def omni_wheel_model(u, x, dt, data):
    """ Omni wheel model """
    l = 0.3  # length from wheel to center of chassis
    r = 0.25  # radius of wheel

    # wheel constraints
    J_1 = np.array([[0, 1, l],
                    [-cos(pi / 6), -sin(pi / 6), data.l],
                    [cos(pi / 6), -sin(pi / 6), data.l]])

    # wheel radius
    J_2 = np.array([[r, 0.0, 0.0], [0.0, r, 0.0], [0.0, 0.0, r]])

    # rotation matrix
    rot = np.array([[cos(x(3)), -sin(x(3)), 0], [sin(x(3)), cos(x(3)), 0],
                    [0, 0, 1]])

    gdot_t = dot(inv(rot), dot(inv(J_1), dot(J_2, u)))
    return x + gdot_t * dt
