from math import cos
from math import sin
from math import pi

import numpy as np
from numpy.linalg import inv


class OmniWheelModel(object):
    """ Omni Wheel Motion Model """

    def __init__(self):
        self.x = np.array([0.0, 0.0, 0.0])  # State vector (x, y, theta)
        self.length = 0.3   # Length from wheel to center of chassis
        self.radius = 0.25  # Radius of wheel

    def omni_wheel_model(self, u, dt):
        """ Omni wheel model """
        # Wheel constraints
        J_1 = np.matrix([[0, 1, self.length],
                        [-cos(pi / 6), -sin(pi / 6), self.length],
                        [cos(pi / 6), -sin(pi / 6), self.length]])

        # Wheel radius
        J_2 = np.matrix([[self.radius, 0.0, 0.0],
                        [0.0, self.radius, 0.0],
                        [0.0, 0.0, self.radius]])

        # Rotation matrix
        R = np.matrix([[cos(self.x(3)), -sin(self.x(3)), 0],
                       [sin(self.x(3)), cos(self.x(3)), 0],
                       [0, 0, 1]])

        gdot_t = inv(R) * inv(J_1) * J_2 * u
        return self.x + gdot_t * dt
