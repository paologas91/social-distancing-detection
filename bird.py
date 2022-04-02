import cv2
import numpy as np


def compute_bird_eye(height, width, pts):
    """

    :param height:
    :param width:
    :param: pts:
    :return:
    """
    img = cv2.imread('first_frame_with_black_stripes.jpg')
    width = img.shape[1]

    # mapping the ROI (region of interest) into a rectangle
    input_pts = np.float32([pts[3], pts[2], pts[1], pts[0]])

    # output_pts = np.float32([[0, 0], [width, 0], [width, 3 * width], [0, 3 * width]])
    output_pts = np.float32([[width / 4, 0], [width, 0], [width * 4, width * 3], [width / 4, width * 3]])
    print("input: ", input_pts)
    print("output: ", output_pts)
    print("height: ", img.shape[0])
    print("width: ", img.shape[1])

    cv2.circle(img, (width, 0), 8, 255, -1)
    cv2.circle(img, (int(width / 4), 0), 8, 255, -1)
    cv2.circle(img, (width, 2 * width), 8, 255, -1)
    cv2.imshow('window', img)
    # Compute the transformation matrix
    filter_m = cv2.getPerspectiveTransform(input_pts, output_pts)
    out = cv2.warpPerspective(img, filter_m, (width, height * 3))

    cv2.imwrite('bird_eye.jpg', out)

    return filter_m, out


def convert_to_bird(centers, filter_m):
    """
    Apply the perpective to the bird's-eye view.
    :param centers:
    :param filter_m:
    return:
    """
    centers = [cv2.perspectiveTransform(np.float32([[center]]), filter_m) for center in centers.copy()]
    centers = [list(center[0, 0]) for center in centers.copy()]
    return centers

