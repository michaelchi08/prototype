#!/usr/bin/env python3
from math import cos
from math import sin
from math import tan

import numpy as np


def ackerman_model(x, u, L, dt):
    """ Ackerman model """
    g1 = x[0] + u[0] * cos(x[2]) * dt
    g2 = x[1] + u[0] * sin(x[2]) * dt
    g3 = x[2] + ((u[0] * tan(x[2])) / L) * dt

    return np.array([g1, g2, g3])
