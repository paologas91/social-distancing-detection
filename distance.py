import sys

import cv2

from bird import convert_to_bird
from functions import compute_distance, mouse_pts, distance_pts

img = None
preview = None
distance_pts = []
initialPoint = (-1, -1)
filled = False
meters = 0.0


def choose_frame_to_draw_distance(filename):
    """
    Converts the video in a compressed mp4 version
    :param filename: The path of the video to convert
    :return:
    """
    cap = cv2.VideoCapture(filename)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        sys.exit()
    frameList=[]
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Display the resulting frame
            frameList.append(frame)
        else:
            break
    cap.release()
    print("Choose the frame where you draw the distance \n Press 'x' to view next frame \n Press 'z' to view previous frame \n Press 's' to save the frame")
    nextFrame=1
    currentFrame=0
    previousFrame=-1
    cv2.imshow("take_frame",frameList[currentFrame])
    while True:
        key = cv2.waitKey(0)
        if  nextFrame<=len(frameList)-1 and key == ord('x'):
            previousFrame = currentFrame
            currentFrame = nextFrame
            nextFrame = nextFrame + 1
            cv2.imshow("take_frame", frameList[currentFrame])
        elif previousFrame>=0 and key == ord('z'):
            nextFrame = currentFrame
            currentFrame = previousFrame
            previousFrame = previousFrame - 1
            cv2.imshow("take_frame", frameList[currentFrame])
        elif key==ord('s'):
            cv2.imwrite("train_frame.jpg",frameList[currentFrame])
            cv2.destroyWindow("take_frame")
            break


def compute_bird_distance(filter_m):
    convert_to_bird_distance = convert_to_bird(distance_pts, filter_m)
    distance_bird = compute_distance(convert_to_bird_distance[0], convert_to_bird_distance[1])
    one_meter_threshold_bird = float(distance_bird) / float(meters)
    return one_meter_threshold_bird


def compute_yolo_distance():
    yolo_distance = compute_distance(distance_pts[0], distance_pts[1])
    one_meter_threshold_yolo = float(yolo_distance) / float(meters)
    return one_meter_threshold_yolo


def recover_two_points():
    """
    Function to recover the four points of the polygon drawn on the image
    :return: The four points
    """

    global filled, img, preview

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