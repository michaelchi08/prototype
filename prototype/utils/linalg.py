import numpy as np


def skew(v):
    """Skew symmetric matrix

    Parameters
    ----------
    v : np.array
        vector of size 3

    Returns
    -------

        Skew symetric matrix (np.matrix)

    """
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A :

    atol :
         (Default value = 1e-13)
    rtol :
         (Default value = 0)

    Returns
    -------


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

    """
    (rows, cols) = A.shape
    for i in rows:
        for j in cols:
            if i == j:
                A[i, j] = abs(A[i, j])
            else:
                x = np.mean([A[i, j], A[j, i]])
                A[i, j] = x
                A[j, i] = x
