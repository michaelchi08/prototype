import cv2
import numpy as np


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
    for f in features:
        cv2.circle(img, tuple(f.pt), 1, color, -1)
    return img


def draw_matches(img_ref, img_cur,
                 fea_ref, fea_cur,
                 match_mask, color=(0, 255, 0)):
    match_img = np.vstack((img_ref, img_cur))
    height_padding = np.array([0, img_cur.shape[0]])

    for i in range(len(match_mask)):
        if match_mask[i]:
            src_pt = tuple(fea_ref[i].pt)
            dst_pt = tuple(fea_cur[i].pt.astype(int) + height_padding)
            cv2.circle(match_img, src_pt, 1, color, -1)
            cv2.circle(match_img, dst_pt, 1, color, -1)
            cv2.line(match_img, src_pt, dst_pt, color, 1)

    return match_img


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
