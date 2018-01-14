import cv2


def draw_points(img, points, color=(0, 255, 0)):
    """Draw points

    Parameters
    ----------
    points : list of np.array
        List of points

    Returns
    -------
    Image with points drawn

    """
    for p in points:
        p = p.ravel()
        cv2.circle(img, (p[0], p[1]), 1, color, -1)
    return img


def draw_keypoints(img, keypoints):
    """Draw keypoints

    Parameters
    ----------
    keypoints : KeyPoint
        List of keypoints

    Returns
    -------
    Image with keypoints drawn

    """
    # Convert to OpenCV KeyPoints
    cv_kps = []
    for kp in keypoints:
        cv_kps.append(kp.as_cv_keypoint())

    # Draw keypoints
    img = cv2.drawKeypoints(img, cv_kps, None, color=(0, 255, 0))
    return img


def draw_features(img, features, color=(0, 255, 0)):
    """Draw features

    Parameters
    ----------
    features : Feature
        List of features

    Returns
    -------
    Image with features drawn

    """
    # # Convert to OpenCV KeyPoints
    # cv_kps = []
    # for f in features:
    #     # cv_kps.append(cv2.KeyPoint(f.pt[0], f.pt[1], f.size))
    #
    # # Draw keypoints
    # img = cv2.drawKeypoints(img, cv_kps, None, color=color)

    for f in features:
        cv2.circle(img, tuple(f.pt[0]), 1, color, -1)
    return img


def convert2cvkeypoints(keypoints):
    """Convert list of KeyPoint to cv2.KeyPoint

    Parameters
    ----------
    keypoints : KeyPoint
        List of keypoints

    Returns
    -------
    List of cv2.KeyPoint

    """
    cv_kps = []
    for kp in keypoints:
        cv_kps.append(kp.as_cv_keypoint())

    return cv_kps
