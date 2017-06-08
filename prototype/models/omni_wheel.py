from math import cos
from math import sin
from math import pi

import numpy as np
from numpy import dot
from numpy.linalg import inv


class OmniWheelModel(object):
    """ Omni Wheel Motion Model """

    def __init__(self):
        self.l = 0.3  # length from wheel to center of chassis
        self.r = 0.25  # radius of wheel

    def omni_wheel_model(self, u, dt):
        """ Omni wheel model """
        # wheel constraints
        J_1 = np.array([[0, 1, self.l],
                        [-cos(pi / 6), -sin(pi / 6), self.l],
                        [cos(pi / 6), -sin(pi / 6), self.l]])

        # wheel radius
        J_2 = np.array([[self.r, 0.0, 0.0],
                        [0.0, self.r, 0.0],
                        [0.0, 0.0, self.r]])

        # rotation matrix
        rot = np.array([[cos(self.x(3)), -sin(self.x(3)), 0],
                        [sin(self.x(3)), cos(self.x(3)), 0],
                        [0, 0, 1]])

        gdot_t = dot(inv(rot), dot(inv(J_1), dot(J_2, u)))
        return self.x + gdot_t * dt
