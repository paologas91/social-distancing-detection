from tqdm import tqdm
import cv2
import numpy as np
import os

img = None
mouse_pts = []
preview = None
initialPoint = (-1, -1)
filled = False


def compute_distance(point_1, point_2):
    """Calculate usual distance.
    param point1:
    param point2:
    return:
    """
    x1, y1 = point_1
    x2, y2 = point_2
    return np.linalg.norm([x1 - x2, y1 - y2])


def recover_four_points(filename, width):
    """

    :param filename:
    :param width:
    :return:
    """

    global mouse_pts, img, filled

    window_name = 'first_frame'
    extension = '.jpg'

    cap = cv2.VideoCapture(filename)
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_lines)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(window_name + extension, frame)
            break
    cap.release()

    img = cv2.imread(window_name + extension)
    color = (0, 0, 0)
    left, right = [int(width / 2)] * 2
    img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=color)

    while not filled:
        # if we are drawing show preview, otherwise the image
        if preview is None:
            cv2.imshow('first_frame', img)
        else:
            cv2.imshow('first_frame', preview)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cv2.imwrite(window_name + '_with_polygon' + extension, img)
    cv2.destroyWindow(window_name)
    return mouse_pts


def draw_lines(event, x, y, flags, param):
    """

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


def ask_to_confirm():
    """
    :return:
    """
    global filled
    window_name = 'first_frame_with_polygon'
    extension = '.jpg'
    first_frame = cv2.imread(window_name + extension)
    cv2.imshow(window_name, first_frame)
    cv2.waitKey(1)
    print('Do you want to confirm the choice? (y/n)')
    answer = input()
    if answer == 'y' or answer == 'Y':
        cv2.destroyWindow(window_name)
        return True
    else:
        cv2.destroyWindow(window_name)
        os.remove(window_name + extension)
        filled = False
    return False


##########################################################################
# ****************    OLD METHODS TO DETECT PEOPLE     *****************
##########################################################################


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


def detect_people_on_frame(model, img, confidence, distance):
    """
    Detect people on a frame and draw the rectangles and lines.
    :param model:
    :param img:
    :param confidence:
    :param distance:
    :return:
    """
    results = model([img[:, :, ::-1]])  # Pass the frame through the model and get the boxes

    xyxy = results.xyxy[0].cpu().numpy()  # xyxy are the box coordinates
    #          x1 (pixels)  y1 (pixels)  x2 (pixels)  y2 (pixels)   confidence        class
    # tensor([[7.47613e+02, 4.01168e+01, 1.14978e+03, 7.12016e+02, 8.71210e-01, 0.00000e+00],
    #         [1.17464e+02, 1.96875e+02, 1.00145e+03, 7.11802e+02, 8.08795e-01, 0.00000e+00],
    #         [4.23969e+02, 4.30401e+02, 5.16833e+02, 7.20000e+02, 7.77376e-01, 2.70000e+01],
    #         [9.81310e+02, 3.10712e+02, 1.03111e+03, 4.19273e+02, 2.86850e-01, 2.70000e+01]])

    xyxy = xyxy[xyxy[:, 4] >= confidence]  # Filter desired confidence
    xyxy = xyxy[xyxy[:, 5] == 0]  # Consider only people
    xyxy = xyxy[:, :4]

    colors = ['green'] * len(xyxy)
    for i in range(len(xyxy)):
        for j in range(i + 1, len(xyxy)):
            # Calculate distance of the centers
            dist, x1, y1, x2, y2 = center_distance(xyxy[i], xyxy[j])
            if dist < distance:
                # If dist < distance, boxes are red and a line is drawn
                colors[i] = 'red'
                colors[j] = 'red'
                img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        # Draw the boxes
        if colors[i] == 'green':
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return img


def detect_people_on_video(model, filename, confidence, distance=90):
    """
    Detect people on a video and draw the rectangles and lines.
    :param model:
    :param filename:
    :param confidence:
    :param distance:
    :return:
    """
    global width, height, fps

    # Capture video
    cap = cv2.VideoCapture(filename)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if os.path.exists('output.avi'):
        os.remove('output.avi')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

    # Iterate through frames and detect people
    vidlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=vidlen) as pbar:
        while cap.isOpened():
            # Read a frame
            ret, frame = cap.read()
            # If it's ok
            if ret:
                frame = detect_people_on_frame(model, frame, confidence, distance)
                # Write new video
                out.write(frame)
                cv2.imshow("Detecting people", frame)
                cv2.waitKey(1)
                pbar.update(1)
            else:
                break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
