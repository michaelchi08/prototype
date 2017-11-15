from math import cos
from math import sin

import numpy as np

import sympy
from sympy import pprint


def two_wheel_2d_model(x, u, dt):
    """Two wheel 2D motion model

    Parameters
    ----------
    x :

    u :

    dt :


    Returns
    -------

    """
    gdot = np.array([[u[0, 0] * cos(x[2, 0]) * dt],
                     [u[0, 0] * sin(x[2, 0]) * dt],
                     [u[1, 0] * dt]])

    return x + gdot


def two_wheel_2d_linearized_model(x, u, dt):
    """Two wheel 2D linearized motion model

    Parameters
    ----------
    x :

    u :

    dt :


    Returns
    -------

    """
    G1 = 1.0
    G2 = 0.0
    G3 = -u[0, 0] * sin(x[2, 0]) * dt

    G4 = 0.0
    G5 = 1.0
    G6 = u[0, 0] * cos(x[2, 0]) * dt

    G7 = 0.0
    G8 = 0.0
    G9 = 1.0

    return np.array([[G1, G2, G3],
                     [G4, G5, G6],
                     [G7, G8, G9]])


def two_wheel_3d_model(x, u, dt):
    """Two wheel 3D motion model

    Parameters
    ----------
    x :

    u :

    dt :


    Returns
    -------

    """
    g1 = x[0, 0] + u[0] * cos(x[3, 0]) * dt
    g2 = x[1, 0] + u[0] * sin(x[3, 0]) * dt
    g3 = x[2, 0] + u[1] * dt
    g4 = x[3, 0] + u[2] * dt

    return np.array([g1, g2, g3, g4])


def two_wheel_2d_deriv():
    """ """
    x1, x2, x3, x4, x5 = sympy.symbols("x1,x2,x3,x4,x5")
    dt = sympy.symbols("dt")

    # x, y, theta, v, omega
    f1 = x1 + x4 * sympy.cos(x3) * dt
    f2 = x2 + x4 * sympy.sin(x3) * dt
    f3 = x3 + x5 * dt
    f4 = x4
    f5 = x5

    F = sympy.Matrix([f1, f2, f3, f4, f5])
    pprint(F.jacobian([x1, x2, x3, x4, x5]))


def two_wheel_3d_deriv():
    """ """
    x1, x2, x3, x4, x5, x6, x7 = sympy.symbols("x1,x2,x3,x4,x5,x6,x7")
    dt = sympy.symbols("dt")

    # x1 - x
    # x2 - y
    # x3 - z
    # x4 - theta
    # x5 - v
    # x6 - omega
    # x7 - vz

    # x, y, z, theta, v, omega, vz
    f1 = x1 + x5 * sympy.cos(x4) * dt
    f2 = x2 + x5 * sympy.sin(x4) * dt
    f3 = x3 + x7 * dt
    f4 = x4 + x6 * dt
    f5 = x5
    f6 = x6
    f7 = x7

    F = sympy.Matrix([f1, f2, f3, f4, f5, f6, f7])
    pprint(F.jacobian([x1, x2, x3, x4, x5, x6, x7]))


def two_wheel_3d_deriv2():
    """ """
    functions = sympy.symbols("f1,f2,f3,f4,f5,f6,f7,f8,f9")
    variables = sympy.symbols("x1,x2,x3,x4,x5,x6,x7,x8,x9")

    f1, f2, f3, f4, f5, f6, f7, f8, f9 = functions
    x1, x2, x3, x4, x5, x6, x7, x8, x9 = variables
    dt = sympy.symbols("dt")

    # x1 - x
    # x2 - y
    # x3 - z

    # x4 - theta
    # x5 - v
    # x6 - vz

    # x7 - omega
    # x8 - a
    # x9 - az
    f1 = x1 + x5 * sympy.cos(x4) * dt
    f2 = x2 + x5 * sympy.sin(x4) * dt
    f3 = x3 + x6 * dt

    f4 = x4 + x7 * dt
    f5 = x5 + x8 * dt
    f6 = x6 + x9 * dt

    f7 = x7
    f8 = x8
    f9 = x9

    F = sympy.Matrix([f1, f2, f3, f4, f5, f6, f7, f8, f9])
    pprint(F.jacobian([x1, x2, x3, x4, x5, x6, x7, x8, x9]))
