from tqdm import tqdm
import cv2
import numpy as np
import os
import torch


def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True, verbose=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model


def convert_video(path):
    # split a given input string into different substrings based on a delimiter and take the first cell of the array
    compressed_path = path.split('.')[0]  # "test"
    compressed_path = 'compressed_' + compressed_path + '.mp4'  # "test.mp4"

    # remove the file if already exists
    if os.path.exists(compressed_path):
        os.remove(compressed_path)

    # Convert video in a specific format
    os.system(f"ffmpeg -i {path} -vcodec libx264 {compressed_path}")


def display_video(filename):

    global random_frame
    counter = 0
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(filename)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            print("frame n° ", counter)
            counter = counter + 1
            if counter == 550:
                random_frame = frame
                cv2.imwrite('random_frame.jpg',random_frame)
            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def center_distance(xyxy1, xyxy2):
    """Calculate the distance of the centers of the boxes."""
    a, b, c, d = xyxy1
    x1 = int(np.mean([a, c]))
    y1 = int(np.mean([b, d]))

    e, f, g, h = xyxy2
    x2 = int(np.mean([e, g]))
    y2 = int(np.mean([f, h]))

    dist = np.linalg.norm([x1 - x2, y1 - y2])
    return dist, x1, y1, x2, y2


def detect_people_on_frame(model, img, confidence, distance):
    """Detect people on a frame and draw the rectangles and lines."""
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


def detect_people_on_video(model, filename, confidence, distance=60):
    """Detect people on a video and draw the rectangles and lines."""

    # Capture video
    cap = cv2.VideoCapture(filename)

    '''Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))'''

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
                cv2.imshow("Capturing", frame)
                cv2.waitKey(1)
                pbar.update(1)
            else:
                break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def recover_four_points(filename):
    global image, mouse_pts

    window_name = 'first_frame'
    extension = '.jpg'
    mouse_pts = []
    cap = cv2.VideoCapture(filename)
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, get_mouse_points)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(window_name + extension, frame)
            break
    cap.release()

    image = cv2.imread(window_name + extension)

    while len(mouse_pts) < 4:
        cv2.imshow(window_name, image)
        cv2.waitKey(1)

    cv2.imwrite(window_name + '_with_polygon' + extension, image)
    cv2.destroyWindow(window_name)


def get_mouse_points(event, x, y, flags, param):
    # Used to mark 4 points on the frame zero of the video that will be warped
    global mouseX, mouseYx
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y

        if len(mouse_pts) != 5:
            cv2.circle(image, (x, y), 3, (0, 255, 255), 5, -1)
            mouse_pts.append((x, y))
            print("Point detected")
            print(mouse_pts)

        if len(mouse_pts) == 4:
            pts = np.array([mouse_pts[0], mouse_pts[1], mouse_pts[3], mouse_pts[2]], np.int32)
            cv2.polylines(image, [pts], True, (0, 255, 255), thickness=4)


def ask_to_confirm():
    window_name = 'first_frame_with_polygon'
    extension = '.jpg'
    first_frame = cv2.imread(window_name + extension)
    cv2.imshow(window_name, first_frame)
    cv2.waitKey(1)
    print('Do you want to confirm the choice? (y/n)')
    answer = input()
    if answer == 'y' or answer == 'Y':
        return True
    else:
        cv2.destroyWindow(window_name)
        os.remove(window_name + extension)
    return False

'''
def compute_bird_eye(filename):
    global width, height, fps

    # Capture video
    cap = cv2.VideoCapture(filename)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frame = cv2.imread('first_frame.jpg')

    # mapping the ROI (region of interest) into a rectangle
    input_pts = np.float32([mouse_pts[0], mouse_pts[1], mouse_pts[1], mouse_pts[3]])
    output_pts = np.float32([[0, 0], [width, 0], [width, 3 * width], [0, 3 * width]])

    print(input_pts)
    print(output_pts)

    # Compute the transformation matrix
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    out = cv2.warpPerspective(frame, M, (width, height))
    print(M)

    cv2.imwrite('bird_eye.jpg', out)
'''

def bird_eye(filename):
    SOLID_BACK_COLOR = (41, 41, 41)
    global fps, width, height

    # Capture video
    cap = cv2.VideoCapture(filename)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale_w = 1.2 / 2
    scale_h = 4 / 2
    frame = cv2.imread('random_frame.jpg')

    # Get perspective
    M, Minv = get_camera_perspective(frame, mouse_pts)
    print(mouse_pts)
    '''pts = np.float32(np.array([mouse_pts[2:]]))
    warped_pt = cv2.perspectiveTransform(pts, M)[0]
    d_thresh = np.sqrt(
        (warped_pt[0][0] - warped_pt[1][0])  2
        + (warped_pt[0][1] - warped_pt[1][1])  2
    )
    bird_image = np.zeros(
        (int(height * scale_h), int(width * scale_w), 3), np.uint8
    )

    bird_image[:] = SOLID_BACK_COLOR
    pedestrian_detect = frame '''
    bird_image = cv2.warpPerspective(frame, M, (width, height))
    cv2.imshow('perspective_window', bird_image)
    cv2.waitKey(0)


def get_camera_perspective(img, src_points):
    IMAGE_H = img.shape[0]
    IMAGE_W = img.shape[1]
    print(IMAGE_W)
    print(IMAGE_H)
    src = np.float32(np.array(src_points))
    # [0,H] bottom-left, [W,H] bottom-right, [H/2,H/2] top-left, [2/3W,H/2] top-right
    # dst = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H3], [0, 0], [IMAGE_W3, 0]])
    dst = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [IMAGE_H/2, IMAGE_H/2], [(2/3*IMAGE_W), IMAGE_H/2]])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    return M, M_inv
