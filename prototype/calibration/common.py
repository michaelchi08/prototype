import cv2


def draw_chessboard_corners(image):
    # Find the chess board corners
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_image, (9, 6), None)

    # Draw image
    if ret is True:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001)
        corners2 = cv2.cornerSubPix(gray_image,
                                    corners,
                                    (11, 11),
                                    (-1, -1),
                                    criteria)
        img = cv2.drawChessboardCorners(image,
                                        (9, 6),
                                        corners2,
                                        ret)

    return img
