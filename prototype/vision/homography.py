import numpy as np


def homography(fp, tp):
    """Find homography H, such that from points `fp` is mapped to to points tp
    using the linear DLT method, points are conditioned automatically.

    Parameters
    ----------
    fp : np.array
        From points
    tp : np.array
        To points

    Returns
    -------

    """
    # Check number of points
    if fp.shape != tp.shape:
        raise RuntimeError("Number of points do not match")

    # Condition points (important for numerical reasons)
    # -- From points --
    m = np.mean(fp[:2], axis=1)
    maxstd = max(np.std(fp[:2], axis=1)) + 1e-9
    C1 = np.diag([1 / maxstd, 1 / maxstd, 1])
    C1[0][2] = -m[0] / maxstd
    C1[1][2] = -m[1] / maxstd
    fp = np.dot(C1, fp)

    # -- To points --
    m = np.mean(tp[:2], axis=1)
    maxstd = max(np.std(tp[:2], axis=1)) + 1e-9
    C2 = np.diag([1 / maxstd, 1 / maxstd, 1])
    C2[0][2] = -m[0] / maxstd
    C2[1][2] = -m[1] / maxstd
    tp = np.dot(C2, tp)

    # Create matrix for linear method, 2 rows for each correspondence pair
    nbr_correspondences = fp.shape[1]
    A = np.zeros((2 * nbr_correspondences, 9))
    for i in range(nbr_correspondences):
        A[2 * i] = [-fp[0][i], -fp[1][i], -1,
                    0, 0, 0,
                    tp[0][i] * fp[0][i], tp[0][i] * fp[1][i], tp[0][i]]
        A[2 * i + 1] = [0, 0, 0,
                        -fp[0][i], -fp[1][i], -1,
                        tp[1][i] * fp[0][i], tp[1][i] * fp[1][i], tp[1][i]]

    # Perform svd
    U, S, V = np.linalg.svd(A)
    H = V[8].reshape((3, 3))  # least squares solution is found at last row

    # Decondition
    H = np.dot(np.linalg.inv(C2), np.dot(H, C1))

    # Normalize and return
    return H / H[2, 2]


def affine_transformation(fp, tp):
    """Find Homography H, affine transformation, such that `tp` is affine
    transform of `fp`.

    Parameters
    ----------
    fp : np.array
        From points
    tp : np.array
        To points

    Returns
    -------

    """
    # Check number of points
    if fp.shape != tp.shape:
        raise RuntimeError("number of points do not match")

    # Condition points
    # --From points--
    m = np.mean(fp[:2], axis=1)
    maxstd = max(np.std(fp[:2], axis=1)) + 1e-9
    C1 = np.diag([1 / maxstd, 1 / maxstd, 1])
    C1[0][2] = -m[0] / maxstd
    C1[1][2] = -m[1] / maxstd
    fp_cond = np.dot(C1, fp)

    # --To points--
    m = np.mean(tp[:2], axis=1)
    C2 = C1.copy()  # must use same scaling for both point sets
    C2[0][2] = -m[0] / maxstd
    C2[1][2] = -m[1] / maxstd
    tp_cond = np.dot(C2, tp)

    # Conditioned points have mean zero, so translation is zero
    A = np.concatenate((fp_cond[:2], tp_cond[:2]), axis=0)
    U, S, V = np.linalg.svd(A.T)

    # Create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]
    tmp2 = np.concatenate((np.dot(C, np.linalg.pinv(B)), np.zeros((2, 1))),
                          axis=1)
    H = np.vstack((tmp2, [0, 0, 1]))

    # Decondition
    H = np.dot(np.linalg.inv(C2), np.dot(H, C1))

    return H / H[2, 2]


def warp_images(img1, img2, tp):
    """Put `img1` in `img2` with an affine transformation such that the corners
    are as close to `tp` as possible, where `tp` are homogeneous and
    counter-clockwise from top left hand corner

    Parameters
    ----------
    img1 :

    img2 :

    tp :


    Returns
    -------

    """
    # Points to warp from
    m, n = img1.shape[:2]
    fp = array([[0, m, m, 0],
                [0, 0, n, n],
                [1, 1, 1, 1]])

    # Compute affine transform and apply
    H = affine_transform(tp, fp)
