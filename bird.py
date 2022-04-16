import cv2
import numpy as np


def compute_bird_eye(pts):
    """
    Computes the bird's eye view of the selected area of the original image bounded by the four points of the polygon
    drawn by the user
    :param pts: The four points of the polygon drawn by the user
    :return:
    """
    img = cv2.imread('first_frame.jpg')
    height = img.shape[0]
    width = img.shape[1]

    # mapping the ROI (region of interest) into a rectangle from bottom left,bottom right,top right,top left
    input_pts = np.float32([pts[0], pts[1], pts[2], pts[3]])

    width_out = width * 2
    if height == 282:
        height_out = height * 4

    else:
        height_out = height * 3
    final_width = width_out + width
    final_height = height_out + height

    output_pts = np.float32([[width, height], [width_out, height], [width_out, height_out], [width, height_out]])

    # Compute the transformation matrix
    filter_m = cv2.getPerspectiveTransform(input_pts, output_pts)
    out = cv2.warpPerspective(img, filter_m, (final_width, final_height))
    cv2.imwrite('bird_eye.jpg', out)
    return filter_m


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
