import cv2
import numpy as np
import os
from system import cls


img = None
mouse_pts = []
preview = None
initialPoint = (-1, -1)
filled = False


def recover_roi_points():
    """
    Function to recover the four points of the polygon drawn on the image
    :return: The four points
    """
    global mouse_pts, img, filled

    # Takes only the name of the file without its extension
    window_name = 'first_frame'

    # Opens the window to start drawing the 4 points
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_polygon)

    # Recovers and saves into the project path the first frame of the selected video
    img = cv2.imread('first_frame.jpg')
    # img = cv2.imread('first_frame.jpg')
    while not filled:
        # if we are drawing show preview, otherwise the image
        if preview is None:
            cv2.imshow('first_frame', img)
        else:
            cv2.imshow('first_frame', preview)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cv2.imwrite(window_name + '_with_polygon.jpg', img)
    cv2.destroyWindow(window_name)
    return mouse_pts


def draw_polygon(event, x, y, flags, param):
    """
    Function to draw the four lines of the polygon
    :param event:
    :param x:
    :param y:
    :param flags:
    :param param:
    :return:
    """
    global initialPoint, preview, mouse_pts, filled

    if len(mouse_pts) == 0:

        if event == cv2.EVENT_LBUTTONDOWN:
            # new initial point and preview is now a copy of the original image
            initialPoint = (x, y)
            preview = img.copy()
            # this will be a point at this point in time
            cv2.line(preview, initialPoint, (x, y), (0, 255, 0), 4)
            mouse_pts.append((x, y))

    elif len(mouse_pts) == 1:
        if event == cv2.EVENT_MOUSEMOVE:
            if preview is not None:
                preview = img.copy()
                cv2.line(preview, mouse_pts[0], (x, y), (0, 255, 0), 4)

        elif event == cv2.EVENT_LBUTTONDOWN:
            if preview is not None:
                preview = None
                cv2.line(img, mouse_pts[0], (x, y), (255, 0, 0), 4)
                mouse_pts.append((x, y))

    elif len(mouse_pts) == 2:

        if event == cv2.EVENT_LBUTTONDOWN:
            if preview is not None:
                preview = None
                cv2.line(img, mouse_pts[1], (x, y), (255, 0, 0), 4)
                mouse_pts.append((x, y))

        elif event == cv2.EVENT_MOUSEMOVE:
            preview = img.copy()
            cv2.line(preview, mouse_pts[1], (x, y), (0, 255, 0), 4)

    elif len(mouse_pts) == 3:
        if event == cv2.EVENT_LBUTTONDOWN:
            if preview is not None:
                preview = None
                cv2.line(img, mouse_pts[2], (x, y), (255, 0, 0), 4)
                mouse_pts.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            preview = img.copy()
            cv2.line(preview, mouse_pts[2], (x, y), (0, 255, 0), 4)

    elif len(mouse_pts) == 4:
        # taking the points starting from bottom left, then bottom right, then top right and top left the points are
        # represented as: the first point on top left, the second point on top right, the third point on bottom right
        # and the fourth point on bottom left
        pts = np.array([mouse_pts[0], mouse_pts[1], mouse_pts[2], mouse_pts[3]], np.int32)
        cv2.polylines(img, [pts], True, (0, 255, 255), thickness=4)
        filled = True


def ask_to_confirm_roi(window_name):
    """
    Asks the user if he wants to accept the selected polygon
    :param window_name:
    :return:
    """
    global filled, mouse_pts

    frame = cv2.imread(window_name)
    cv2.imshow(window_name, frame)
    cv2.waitKey(1)
    print('Do you want to confirm the choice? (y/n)')
    answer = input()
    if answer == 'y' or answer == 'Y':
        cv2.destroyWindow(window_name)
        cls()
        return True
    else:
        filled = False
        mouse_pts = []

        cv2.destroyWindow(window_name)
        os.remove(window_name)
        return False
