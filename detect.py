import cv2
import numpy as np
from tqdm import tqdm

from bird import convert_to_bird, compute_bird_eye
from distance import compute_distance_from_set_point
from functions import compute_distance
from video import saveVideo


def detect_people_on_frame(model, frame, confidence, height, width, pts, filename, frame_number, distance):
    """
    Detect people on a frame and draw the rectangles and lines.
    :param model:
    :param frame:
    :param confidence:
    :param distance:
    :param height:
    :param width:
    :param pts:
    :param filename
    :param frame_number
    :return:
    """
    n_violations = 0
    # Pass the frame through the model and get the boxes
    results = model([frame[:, :, ::-1]])

    img = cv2.imread('first_frame_with_polygon.jpg')
    height_1 = img.shape[0]
    width_1 = img.shape[1]

    # Return a new array of given shape and type, filled with zeros.
    bird_eye_background = np.zeros((height_1, width_1, 3), np.uint8)
    print("dimensioni bird_eye_background: ", bird_eye_background.shape)
    bird_eye_background[:, :, :] = 0

    xyxy = results.xyxy[0].cpu().numpy()  # xyxy are the box coordinates
    #          x1 (pixels)  y1 (pixels)  x2 (pixels)  y2 (pixels)   confidence        class
    # tensor([[7.47613e+02, 4.01168e+01, 1.14978e+03, 7.12016e+02, 8.71210e-01, 0.00000e+00],
    #         [1.17464e+02, 1.96875e+02, 1.00145e+03, 7.11802e+02, 8.08795e-01, 0.00000e+00],
    #         [4.23969e+02, 4.30401e+02, 5.16833e+02, 7.20000e+02, 7.77376e-01, 2.70000e+01],
    #         [9.81310e+02, 3.10712e+02, 1.03111e+03, 4.19273e+02, 2.86850e-01, 2.70000e+01]])

    xyxy = xyxy[xyxy[:, 4] >= confidence]  # Filter desired confidence
    xyxy = xyxy[xyxy[:, 5] == 0]  # Consider only people
    xyxy = xyxy[:, :4]

    # Number of rows of xyxy correspond to the number of person inside each frame
    shape = np.shape(xyxy)

    # Calculate the centers of the bottom of the boxes
    centers = []
    for x1, y1, x2, y2 in xyxy:
        center = [np.mean([x1, x2]), y2]
        centers.append(center)

    filter_m, warped = compute_bird_eye(pts)

    if frame_number == 1:
        distance = compute_distance_from_set_point(filter_m)
        print('distance =', distance)

    # Convert to bird so we can calculate the usual distance
    bird_centers = convert_to_bird(centers, filter_m)
    # warped = cv2.resize(bird_eye_background, (width, height))

    colors = ['green'] * len(bird_centers)
    shift = int(width/2)

    for i in range(len(bird_centers)):
        for j in range(i + 1, len(bird_centers)):
            # Calculate distance of the centers
            dist = compute_distance(bird_centers[i], bird_centers[j])

            if dist < distance:
                # If dist < distance, boxes are red and a line is drawn
                colors[i] = 'red'
                colors[j] = 'red'
                x1, y1 = bird_centers[i]
                x2, y2 = bird_centers[j]

                print(int(x1), int(y1), int(x2), int(y2))

                # Increments the number of violations
                n_violations = n_violations + 1

                # Draws a red line between the two persons which are violating the distance
                warped= cv2.line(warped,
                                               (int(x1), int(y1)),
                                               (int(x2), int(y2)),
                                               (0, 0, 255), 2)

    for i, bird_center in enumerate(bird_centers):
        if colors[i] == 'green':
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        x, y = bird_center
        x = int(x)
        y = int(y)

        # TODO: Modify the radius of the circle based on video resolution
        #bird_eye_background = cv2.circle(bird_eye_background, (x, y), 8, color, -1)
        warped = cv2.circle(warped, (x, y), 8, color, -1)
    warped_flip = cv2.flip(warped, 0)

    # Concat the black bird-eye image with the frame
    warped_flip = cv2.resize(warped_flip, (width, height))
    warped_flip = cv2.hconcat([warped_flip, frame])


    # Display the number of people in the frame
    cv2.putText(img=warped_flip,
                text="Number of people: " + str(shape[0]),
                org=(335, 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=2)

    # Display the number of violations in the frame
    cv2.putText(img=warped_flip,
                text="Number of violations: " + str(n_violations),
                org=(335, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=2)

    return centers, bird_centers, warped_flip, distance


def detect_people_on_video(model, filename, fps, height, width, pts, confidence):
    """
    Detect people on a video and draw the rectangles and lines.
    :param model:
    :param filename:
    :param fps:
    :param height:
    :param width:
    :param pts:
    :param confidence:
    :return:
    """

    # Capture video
    cap = cv2.VideoCapture(filename)
    frame_number = 0
    distance = 0
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi', fourcc, fps, (width * 2, height))

    out = saveVideo(fourcc, fps, width, height)

    # Iterate through frames and detect people
    vidlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=vidlen) as pbar:
        while cap.isOpened():
            # Read a frame
            ret, frame = cap.read()
            # If it's ok
            if ret:
                frame_number = frame_number + 1
                centers, bird_centers, frame, distance = detect_people_on_frame(model,
                                                                                frame,
                                                                                confidence,
                                                                                height,
                                                                                width,
                                                                                pts, filename, frame_number, distance)
                print('frame n°', frame_number)
                print('#####centers####', centers)
                print('####bird_centers####', bird_centers)

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
