

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
