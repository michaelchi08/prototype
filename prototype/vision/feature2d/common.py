import cv2


def draw_keypoints(img, keypoints):
    """Draw keypoints"""
    # Convert to OpenCV KeyPoints
    cv_kps = []
    for kp in keypoints:
        cv_kps.append(kp.as_cv_keypoint())

    # Draw keypoints
    img = cv2.drawKeypoints(img, cv_kps, None, color=(0, 255, 0))
    return img


def draw_features(img, features):
    """Draw features"""
    # Convert to OpenCV KeyPoints
    cv_kps = []
    for f in features:
        cv_kps.append(cv2.KeyPoint(f.pt[0], f.pt[1], f.size))

    # Draw keypoints
    img = cv2.drawKeypoints(img, cv_kps, None, color=(0, 255, 0))
    return img


def convert2cvkeypoints(keypoints):
    """Convert list of KeyPoint to cv2.KeyPoint"""
    cv_kps = []
    for kp in keypoints:
        cv_kps.append(kp.as_cv_keypoint())

    return cv_kps
