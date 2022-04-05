import csv

import cv2
import numpy as np

from bird import convert_to_bird
from functions import compute_distance, ask_to_confirm

img = None
mouse_pts = []
preview = None
initialPoint = (-1, -1)
filled = False

def get_distance_from_video(filename):
    """
    Converts the video in a compressed mp4 version
    :param path: The path of the video to convert
    :return:
    """

    answer=False
    while not answer:
        print("filename:", filename)
        cap = cv2.VideoCapture(filename)

        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error opening video stream or file")
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                # Display the resulting frame
                cv2.imshow(filename, frame)
                cv2.waitKey(0)
                # Press Q on keyboard to  exit
                if cv2.waitKey(0) & 0xFF == ord('y'):
                    cv2.imwrite("train_frame.jpg",frame)
                    cv2.destroyWindow(filename)
                    break
            # Break the loop
            else:
                break
        # When everything done, release the video capture object

        distance_points = recover_two_points()
        cap.release()
        answer = ask_to_confirm('train_frame_with_line.jpg')
    return distance_points

def compute_distance_from_set_point(filename,filter_m):
    distance_points = get_distance_from_video(filename)
    convert_to_bird_distance = convert_to_bird(distance_points, filter_m)
    distance_bird = compute_distance(convert_to_bird_distance[0], convert_to_bird_distance[1])
    return distance_bird



def recover_two_points():
    """
    Function to recover the four points of the polygon drawn on the image
    :return: The four points
    """
    global mouse_pts, img, filled

    # Takes only the name of the file without its extension
    window_name = 'train_frame'

    # Opens the window to start drawing the 4 points
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_distance)

    # Recovers and saves into the project path the first frame of the selected video
    img = cv2.imread('train_frame.jpg')
    while not filled:
        # if we are drawing show preview, otherwise the image
        if preview is None:
            cv2.imshow('train_frame', img)
        else:
            cv2.imshow('train_frame', preview)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cv2.imwrite(window_name + '_with_line.jpg', img)
    cv2.destroyWindow(window_name)
    return mouse_pts


def draw_distance(event, x, y, flags, param):
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
        # taking the points starting from bottom left, then bottom right, then top right and top left the points are
        # represented as: the first point on top left, the second point on top right, the third point on bottom right
        # and the fourth point on bottom left
        pts = np.array([mouse_pts[0], mouse_pts[1]], np.int32)
        filled = True



