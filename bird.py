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
    heightMax= img.shape[0]
    widthMax = img.shape[1]

    # mapping the ROI (region of interest) into a rectangle from bottom left,bottom right,top right,top left
    input_pts = np.float32([pts[0], pts[1], pts[2], pts[3]])


    #output_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    if heightMax==228:
        output_pts=np.float32([[pts[0][0], pts[2][1]*2],[pts[1][0], pts[2][1]*2],pts[1],pts[0]])
        heightMax=heightMax*3
    else:
        output_pts = np.float32([[pts[0][0], pts[2][1]],[pts[1][0], pts[2][1]],pts[1],pts[0]])
        heightMax = heightMax * 2


    print("input:",input_pts)
    print ("output:",output_pts)

    # Compute the transformation matrix
    filter_m = cv2.getPerspectiveTransform(input_pts, output_pts)

    out = cv2.warpPerspective(img, filter_m, (widthMax, heightMax))


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
