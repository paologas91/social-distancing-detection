import sys
import cv2
import os
import numpy as np
from bird import convert_to_bird
from system import cls

img = None
preview = None
distance_pts = []
initialPoint = (-1, -1)
filled = False
meters = 0.0


def compute_distance(point_1, point_2):
    """
    Calculate usual distance
    :param point_1: First point
    :param point_2: Second point
    :return: The distance
    """
    x1, y1 = point_1
    x2, y2 = point_2
    return np.linalg.norm([x1 - x2, y1 - y2])


def center_distance(xyxy1, xyxy2):
    """
    Calculate the distance of the centers of the boxes.
    :param xyxy1:
    :param xyxy2:
    :return:
    """
    a, b, c, d = xyxy1
    x1 = int(np.mean([a, c]))
    y1 = int(np.mean([b, d]))

    e, f, g, h = xyxy2
    x2 = int(np.mean([e, g]))
    y2 = int(np.mean([f, h]))

    dist = np.linalg.norm([x1 - x2, y1 - y2])
    return dist, x1, y1, x2, y2


def choose_distance_frame(filename):
    """

    :param filename:
    :return:
    """
    cap = cv2.VideoCapture(filename)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        sys.exit()
    frame_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Display the resulting frame
            frame_list.append(frame)
        else:
            break
    cap.release()
    print("Choose the frame on which you will draw the distance \n Press 'x' to display the next frame \n Press 'z' "
          "to display the previous frame \n Press 's' to choose the frame")
    next_frame = 1
    current_frame = 0
    previous_frame = -1
    cv2.imshow("Choose the frame on which you will draw the distance", frame_list[current_frame])
    while True:
        key = cv2.waitKey(0) & 0xFF

        if next_frame <= len(frame_list) - 1 and key == ord('x'):

            previous_frame = current_frame
            current_frame = next_frame
            next_frame = next_frame + 1
            cv2.imshow("Choose the frame on which you will draw the distance", frame_list[current_frame])

        elif previous_frame >= 0 and key == ord('z'):

            next_frame = current_frame
            current_frame = previous_frame
            previous_frame = previous_frame - 1
            cv2.imshow("Choose the frame on which you will draw the distance", frame_list[current_frame])

        elif key == ord('s'):

            cv2.imwrite("distance_frame.jpg", frame_list[current_frame])
            cv2.destroyWindow("Choose the frame on which you will draw the distance")
            break


def compute_bird_distance(filter_m):
    """

    :param filter_m:
    :return:
    """
    convert_to_bird_distance = convert_to_bird(distance_pts, filter_m)
    distance_bird = compute_distance(convert_to_bird_distance[0], convert_to_bird_distance[1])
    one_meter_threshold_bird = float(distance_bird) / float(meters)

    return one_meter_threshold_bird


def compute_yolo_distance():
    """

    :return:
    """
    yolo_distance = compute_distance(distance_pts[0], distance_pts[1])
    one_meter_threshold_yolo = float(yolo_distance) / float(meters)

    return one_meter_threshold_yolo


def recover_distance_points():
    """
    Function to recover the two points of the distance
    :return: The two points
    """

    global filled, img, preview

    # Takes only the name of the file without its extension
    window_name = 'distance_frame'

    # Opens the window to start drawing the 4 points
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_distance)

    # Recovers and saves into the project path the first frame of the selected video
    img = cv2.imread('distance_frame.jpg')
    while not filled:
        # if we are drawing show preview, otherwise the image
        if preview is None:
            cv2.imshow('distance_frame', img)
        else:
            cv2.imshow('distance_frame', preview)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cv2.imwrite(window_name + '_with_line.jpg', img)
    cv2.destroyWindow(window_name)
    return distance_pts


def insert_distance_in_meters():
    global meters

    print("Insert the value of the distance in meters: ")
    meters = input()


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
    global initialPoint, distance_pts, preview, filled, img

    if len(distance_pts) == 0:

        if event == cv2.EVENT_LBUTTONDOWN:
            # new initial point and preview is now a copy of the original image
            initialPoint = (x, y)
            preview = img.copy()
            # this will be a point at this point in time
            cv2.line(preview, initialPoint, (x, y), (0, 255, 0), 4)
            distance_pts.append((x, y))

    elif len(distance_pts) == 1:
        if event == cv2.EVENT_MOUSEMOVE:
            if preview is not None:
                preview = img.copy()
                cv2.line(preview, distance_pts[0], (x, y), (0, 255, 0), 4)

        elif event == cv2.EVENT_LBUTTONDOWN:
            if preview is not None:
                preview = None
                cv2.line(img, distance_pts[0], (x, y), (255, 0, 0), 4)
                distance_pts.append((x, y))

    elif len(distance_pts) == 2:
        filled = True


def ask_to_confirm_distance(window_name):
    """

    :return:
    """
    global filled, distance_pts

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
        distance_pts = []

        cv2.destroyWindow(window_name)
        os.remove(window_name)
        return False
