import cv2
import numpy as np


def compute_bird_eye(height, width, pts):
    """
    Computes the bird's eye view of the selected area of the original image bounded by the four points of the polygon
    drawn by the user
    :param height:
    :param width:
    :param pts: The four points of the polygon drawn by the user
    :return:
    """
    img = cv2.imread('first_frame.jpg')
    img_with_polygon = cv2.imread('first_frame_with_polygon.jpg')
    # width = img.shape[1]

    # mapping the ROI (region of interest) into a rectangle
    input_pts = np.float32([pts[0], pts[3], pts[2], pts[1]])

    # output_pts = np.float32([[0, 0], [width, 0], [width, 3 * width], [0, 3 * width]])
    output_pts = np.float32([[width / 4, 0], [width, 0], [width * 4, width * 3], [width / 4, width * 3]])
    print("input_pts: ", input_pts)
    print("output_pts: ", output_pts)
    print("height of first_frame.jpg: ", img.shape[0])
    print("width of first_frame.jpg: ", img.shape[1])

    cv2.circle(img_with_polygon, (width, 0), 8, 255, -1)
    cv2.circle(img_with_polygon, (int(width / 4), 0), 8, 255, -1)
    cv2.circle(img_with_polygon, (width, 2 * width), 8, 255, -1)

    # Compute the transformation matrix
    filter_m = cv2.getPerspectiveTransform(input_pts, output_pts)
    out = cv2.warpPerspective(img, filter_m, (width, height * 3))

    cv2.imwrite('bird_eye.jpg', out)

    return filter_m, out


def convert_to_bird(centers, filter_m):
    """
    Apply the perspective to the bird's-eye view.
    :param centers:
    :param filter_m:
    return:
    """
    centers = [cv2.perspectiveTransform(np.float32([[center]]), filter_m) for center in centers.copy()]
    centers = [list(center[0, 0]) for center in centers.copy()]
    return centers

