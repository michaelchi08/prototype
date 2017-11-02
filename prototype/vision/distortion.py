import numpy as np


def radtan(k_1, k_2, k_3, t_1, t_2):
    u  = x / z
    v  = y / z
    r = u**2 + v**2
    d_r = 1 + k_1 * r + k_2 * r**2 + k_3 * r**3
    d_t = np.array([[2 * u * v * t_1 + (r + 2 * u**2) * t_2]
                    [2 * u * v * t_2 + (r + 2 * v**2) * t_1]])

    intrinsics = np.matrix([[f_x, 0.0], [0.0, f_y]])
    principle = np.matrix([[p_x], [p_y]])
    principle + intrinsics * d_r * np.array([[u], [v]]) + d_t


def equidistance():
    pass
