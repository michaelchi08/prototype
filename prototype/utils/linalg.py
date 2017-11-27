import numpy as np
from numpy import dot
from numpy import eye as I
from numpy.linalg import norm

import scipy


def skew(w):
    """Skew symmetric matrix

    Parameters
    ----------
    w : np.array
        vector of size 3

    Returns
    -------

        Skew symetric matrix (np.matrix)

    """
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def skewsq(w):
    """Skew symmetric matrix squared

    Parameters
    ----------
    w : np.array
        vector of size 3

    Returns
    -------

        Squared skew symetric matrix (np.matrix)

    """
    return (dot(w, w.T) - dot(norm(w)**2, I(3)))


def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : np.array
        Matrix to form null space from
    atol : float
         (Default value = 1e-13)
    rtol : float
         (Default value = 0)

    Returns
    -------
    ns : np.array
        Nullspace of A

    """

    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def enforce_psd(A):
    """Enforce Positive Semi-definite

    Parameters
    ----------
    A : np.array
        Matrix to ensure positive semi-definite

    Returns
    -------
    A : np.array
        Positive semi-definite matrix

    """
    rows, cols = A.shape
    for i in range(rows):
        for j in range(cols):
            if i == j:
                A[i, j] = abs(A[i, j])
            else:
                x = 0.5 * (A[i, j] + A[j, i])
                A[i, j] = x
                A[j, i] = x

    return A
