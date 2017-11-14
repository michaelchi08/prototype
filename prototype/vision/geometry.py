import numpy as np


def eight_point(x1, x2):
    """Computes the fundamental matrix from corresponding points
    x1 and x2 as a 3xN matrix (np.array) using the normalized 8 point algorithm.
    each row is constructed as:

        [x' * x, x' * y, x', y' * x, y' * y, y', x, y, 1]

    Parameters
    ----------
    x1 : np.array
        Homogenouse pixel location from first camera
    x2 : np.array
        Homogenouse pixel location from second camera

    Returns
    -------

        Fundamental matrix (np.array)

    """
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don’t match.")

    # Build matrix for equations
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [x1[0, i]*x2[0, i], x1[0, i]*x2[1, i], x1[0, i]*x2[2, i],
                x1[1, i]*x2[0, i], x1[1, i]*x2[1, i], x1[1, i]*x2[2, i],
                x1[2, i]*x2[0, i], x1[2, i]*x2[1, i], x1[2, i]*x2[2, i]]

    # Compute linear least square solution
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # Constrain F
    # Make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))

    return F


def triangulate_point(x1, x2, P1, P2):
    """Triangulate a single 3D point using a pair of homogenous pixel
    observations x1, x2 and two camera views encapsulated by two camera matrices
    P1, P2 via the linear least squares method

    Parameters
    ----------
    x1 : np.array
        Homogenouse pixel location from first camera
    x2 : np.array
        Homogenouse pixel location from second camera
    P1 : np.array
        Camera 1 matrix (P = K [R | t])
    P2 : np.array
        Camera 2 matrix (P = K [R | t])

    Returns
    -------

        Homogeneous 3D point (np.array)

    """
    # Make inputs x1 and x2 homogeneous if not already
    if len(x1) == 2:
        x1 = np.array([x1[0], x1[1], 1.0])
    if len(x2) == 2:
        x2 = np.array([x2[0], x2[1], 1.0])

    # Build the A matrix for linear least squares
    A = np.zeros((6, 6))
    A[:3, :4] = P1
    A[3:, :4] = P2
    A[:3, 4] = -x1
    A[3:, 5] = -x2

    # Perform linear least squares using SVD and get result
    try:
        U, S, V = np.linalg.svd(A)
        X = V[-1, :4]
        return (X / X[3])
    except:
        return None


def triangulate(x1, x2, P1, P2):
    """Triangulate a set of 3D points from two sets of feature vectors x1, x2
    and two camera views encapsulated by two camera matrices P1, P2 via the
    linear least squares method

    Parameters
    ----------
    x1 : np.array
        Observations in the first camera as 2xN matrix
    x2 : np.array
        Observations in the second camera as 2xN matrix
    P1 : np.array
        Camera 1 projection matrix
    P2 : np.array
        Camera 2 projection matrix

    Returns
    -------
    X : np.array
        Homogeneous 3D points as a 4xN matrix (np.array)

    """
    # Pre-check
    if x1.shape[0] != 2 or x2.shape[0] != 2:
        raise RuntimeError("Input is not a 2xN matrix")
    elif x1.shape != x2.shape:
        raise RuntimeError("x1 and x2 are not same size!")

    n = x1.shape[1]
    X = [triangulate_point(x1[:, i], x2[:, i], P1, P2) for i in range(n)]
    X = np.array(X).T

    return X
