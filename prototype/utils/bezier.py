import numpy as np
from math import pi
from math import cos
from math import sin
from math import factorial as fac


def binomial(x, y):
    try:
        binom = fac(x) // fac(y) // fac(x - y)
    except ValueError:
        binom = 0
    return binom


def bezier_quadratic(P0, C0, P1, t):
    """Quadratic Bezier curve

    .. math:
        \begin{align*}
            Q_0 &= (1 - t) P_0 + t C_0 , t \in [0,1] \\
            Q_1 &= (1 - t) C_0 + t P_1, t \in [0,1] \\
            R_0 &= (1 - t) Q_0 + t Q_1, t \in [0,1]
        \end{align*}

    Parameters
    ----------
    P0 : np.array
        Start anchor point
    C0 : np.array
        Control point
    P1 : np.array
        End anchor point
    t : float
        Parameter

    Returns
    -------
    r : np.array
        Position on bezier curve at t

    """
    Q0 = (1 - t) * P0 + t * C0
    Q1 = (1 - t) * C0 + t * P1
    r = (1 - t) * Q0 + t * Q1

    return r


def bezier_cubic(P0, C0, C1, P1, t):
    """Cubic Bezier curve

    .. math:
        \begin{equation}
            s(t) &= (1 - t)^3 A_0
                + 3(1 - t)^2 t C_0
                + 3(1 - t) t^2 C_1
                + t^3 A_1, t \in [0,1] \\
        \end{equation}

    Parameters
    ----------
    P0 : np.array
        Start anchor point
    C0 : np.array
        First control point
    C1 : np.array
        Second control point
    P1 : np.array
        End anchor point
    t : float
        Parameter

    Returns
    -------
    s : np.array
        Position on bezier curve at t

    """
    s = (1 - t)**3 * P0
    s += 3 * (1 - t)**2 * t * C0
    s += 3 * (1 - t) * t**2 * C1
    s += t**3 * P1

    return s


def bezier_cubic_velocity(P0, C0, C1, P1, t):
    """Velocity of cubic Bezier curve

    .. math:
        \begin{equation}
            v_0(t) &= 3 \left(
                (1-t)^2 (C_0-A_0)
                + 2 (1-t)t(C_1-C_0)
                + t^2(A_1-C_1)
            \right), t\in [0,1]
        \end{equation}

    Parameters
    ----------
    P0 : np.array
        Start anchor point
    C0 : np.array
        First control point
    C1 : np.array
        Second control point
    P1 : np.array
        End anchor point
    t : float
        Parameter

    Returns
    -------
    v : np.array
        Velocity on bezier curve at t

    """
    v = (1 - t)**2 * (C0 - P0)
    v += 2 * (1 - t) * t * (C1 - C0)
    v += t**2 * (P1 - C1)
    v = 3 * v
    return v


def bezier_cubic_acceleration(P0, C0, C1, P1, t):
    """Acceleration of cubic Bezier curve

    .. math:
        \begin{equation}
            a(t) &= 6\left(
                (1 - t)(C_1 - 2 C_0 + P_0)
                + t (P_1 - 2C_1 + C_0)
            \right), t \in [0, 1]
        \end{equation}

    Parameters
    ----------
    P0 : np.array
        Start anchor point
    C0 : np.array
        First control point
    C1 : np.array
        Second control point
    P1 : np.array
        End anchor point
    t : float
        Parameter

    Returns
    -------
    a : np.array
        Acceleration on bezier curve at t

    """
    a = (1 - t) * (C1 - 2 * C0 + P0)
    a += t * (P1 - 2 * C1 + C0)
    a = 6 * a
    return a


def bezier(points, t):
    """Explicit definition of a bezier curve

    .. math:
        \begin{align}
            \mathbf{B}(t) &=
                \sum_{i=0}^n {n\choose i}(1 - t)^{n - i}t^i\mathbf{P}_i \\
                &= (1 - t)^n\mathbf{P}_0
                    + {n\choose 1}(1 - t)^{n - 1} t \mathbf{P}_1
                    + \cdots
                    + {n\choose n - 1}(1 - t)t^{n - 1} \mathbf{P}_{n - 1}
                    + t^n\mathbf{P}_n && 0 \leqslant t \leqslant 1
        \end{align}

    Parameters
    ----------
    points : np.array
        Bezier curve control points in order
    t : float
        Parameter

    Returns
    -------
    s : np.array
        Position on bezier curve at t

    """
    n = len(points) - 1

    result = np.array([0.0, 0.0, 0.0])
    for i in range(0, n + 1):
        binomial_term = binomial(n, i)
        polynomial_term = (1 - t)**(n - i) * t**i
        weight = points[i]
        result = result + binomial_term * polynomial_term * weight

    return result


def decasteljau(points, t):
    """De Casteljau's algorithm

    A recursive method to evaluate polynomials in Bernstein form or Bezier
    curves. This algorithm can also be used to split a single Bezier curve into
    two at an arbitrary parameter value.

    Parameters
    ----------
    points : np.array
        Bezier curve control points in order
    t : float
        Parameter

    Returns
    -------
    s : np.array
        Position on bezier curve at t

    """
    if len(points) == 1:
        return points[0]

    new_points = []
    for i in range(len(points) - 1):
        new_points.append((1 - t) * points[i] + t * points[i + 1])

    s = decasteljau(new_points, t)
    return s


def bezier_derivative(points, t, order):
    """Derivative of an arbitrary order Bezier curve

    Source: https://pomax.github.io/bezierinfo/#derivatives

    Parameters
    ----------
    points : np.array
        Bezier curve control points in order
    t : float
        Parameter

    Returns
    -------
    s : np.array
        Position on bezier curve at t

    """
    n = len(points) - 1
    k = n - 1

    new_points = []
    for i in range(0, k + 1):
        binomial_term = binomial(k, i)
        polynomial_term = (1 - t)**(k - i) * t**i
        derivative_weight = n * (points[i + 1] - points[i])
        new_points.append(binomial_term * polynomial_term * derivative_weight)

    if order == 1:
        return np.sum(new_points, axis=0)
    else:
        return bezier_derivative(new_points, t, order - 1)


def bezier_tangent(points, t):
    point = bezier_derivative(points, t, 1)
    d = np.linalg.norm(point)
    tangent = point / d

    return tangent


def bezier_normal(points, t):
    tangent = bezier_tangent(points, t)

    theta = pi / 2.0
    R = np.array([[cos(theta), -sin(theta), 0.0],
                  [sin(theta), cos(theta), 0.0],
                  [0.0, 0.0, 1.0]])
    normal = np.dot(R, tangent)

    return normal
