import cv2
import numpy as np
from tqdm import tqdm
from bird import convert_to_bird, compute_bird_eye
from distance import compute_bird_distance, compute_yolo_distance, compute_distance, center_distance
from video import save_video


def detect_people_on_frame(model, sdd_frame, confidence, height, width, pts, frame_number, one_meter_threshold_bird,
                           one_meter_threshold_yolo):
    """
    Detect people on a frame and draw the rectangles and lines
    :param one_meter_threshold_bird:
    :param one_meter_threshold_yolo:
    :param model:
    :param sdd_frame:
    :param confidence:
    :param height:
    :param width:
    :param pts:
    :param frame_number
    :return:
    """
    sdd_violations = 0
    yolo_violations = 0

    # Pass the frame through the model and get the boxes
    results = model([sdd_frame[:, :, ::-1]])

    # Return a new array of given shape and type, filled with zeros.
    bird_eye_background = np.zeros((height, width, 3), np.uint8)
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

    # Calculate the yolo_centers of the bottom of the boxes
    yolo_centers = []
    for x1, y1, x2, y2 in xyxy:
        center = [np.mean([x1, x2]), y2]
        yolo_centers.append(center)

    # copy the sdd_frame to save the reference
    yolo_frame = sdd_frame.copy()

    filter_m, bird_eye_frame = compute_bird_eye(pts)

    if frame_number == 1:
        one_meter_threshold_bird = compute_bird_distance(filter_m)
        one_meter_threshold_yolo = compute_yolo_distance()

    # Convert to bird so we can calculate the usual distance
    bird_centers = convert_to_bird(yolo_centers, filter_m)

    # initialize bird_colors array
    bird_colors = ['green'] * len(bird_centers)

    # initialize bird_colors array
    yolo_colors = ['green'] * len(yolo_centers)

    for i in range(len(bird_centers)):
        for j in range(i + 1, len(bird_centers)):

            # Calculate distance of the yolo_centers
            dist = compute_distance(bird_centers[i], bird_centers[j])
            dist_1, x1_1, y1_1, x2_1, y2_1 = center_distance(xyxy[i], xyxy[j])

            print("\n")
            print("Bird distance: ", dist)
            print("Bird Threshold:", one_meter_threshold_bird)

            print("Yolo distance: ", dist_1)
            print("Yolo Threshold:", one_meter_threshold_yolo)
            print("\n")

            if dist < one_meter_threshold_bird:
                # If dist < distance, boxes are red and a line is drawn
                bird_colors[i] = 'red'
                bird_colors[j] = 'red'

                x1, y1 = bird_centers[i]
                x2, y2 = bird_centers[j]

                # Increments the number of violations
                sdd_violations = sdd_violations + 1

                # Draws a red line in the bird_eye frame between the two persons which are violating the distance
                bird_eye_frame = cv2.line(bird_eye_frame,
                                          (int(x1), int(y1)),
                                          (int(x2), int(y2)),
                                          (0, 0, 255), 2)

                # Draws a red line in the sdd_frame between the two persons which are violating the distance
                sdd_frame = cv2.line(sdd_frame,
                                     (int(x1_1), int(y1_1)),
                                     (int(x2_1), int(y2_1)),
                                     (0, 0, 255), 2)

            if dist_1 < one_meter_threshold_yolo:
                # If dist < distance, boxes are red and a line is drawn
                yolo_colors[i] = 'red'
                yolo_colors[j] = 'red'

                # Increments the number of violations
                yolo_violations = yolo_violations + 1

                # Draws a red line in the yolo_frame between the two persons which are violating the distance
                yolo_frame = cv2.line(yolo_frame,
                                      (int(x1_1), int(y1_1)),
                                      (int(x2_1), int(y2_1)),
                                      (0, 0, 255), 2)

    # draw the circles in the bird_eye frame
    for i, bird_center in enumerate(bird_centers):

        if bird_colors[i] == 'green':
            bird_color = (0, 255, 0)
        else:
            bird_color = (0, 0, 255)

        x, y = bird_center
        x = int(x)
        y = int(y)

        # TODO: Modify the radius of the circle based on video resolution
        bird_eye_frame = cv2.circle(bird_eye_frame, (x, y), 8, bird_color, -1)

    # draw the rectangles in yolo and sdd frames
    for i, (x1, y1, x2, y2) in enumerate(xyxy):

        if bird_colors[i] == 'green':
            bird_color = (0, 255, 0)
        else:
            bird_color = (0, 0, 255)

        if yolo_colors[i] == 'green':
            yolo_color = (0, 255, 0)
        else:
            yolo_color = (0, 0, 255)

        sdd_frame = cv2.rectangle(sdd_frame, (int(x1), int(y1)), (int(x2), int(y2)), bird_color, 2)
        yolo_frame = cv2.rectangle(yolo_frame, (int(x1), int(y1)), (int(x2), int(y2)), yolo_color, 2)

    # Concat the yolo, bird-eye and ssd frames into one
    bird_eye_frame = cv2.resize(bird_eye_frame, (width, height))
    bird_eye_frame = cv2.hconcat([yolo_frame, bird_eye_frame])
    bird_eye_frame = cv2.hconcat(([bird_eye_frame, sdd_frame]))

    # add border for titles and description
    color = (0, 0, 0)
    bottom, up = [50] * 2
    bird_eye_frame = cv2.copyMakeBorder(bird_eye_frame, bottom, up, 0, 0, cv2.BORDER_CONSTANT, value=color)

    # display titles and counter
    add_text(bird_eye_frame, height, width, shape, sdd_violations, yolo_violations)

    return yolo_centers, bird_centers, bird_eye_frame, one_meter_threshold_bird, one_meter_threshold_yolo


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
    one_meter_threshold_bird = 0
    one_meter_threshold_yolo = 0

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = save_video(fourcc, fps, width, height)

    # Iterate through frames and detect people
    vidlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=vidlen) as pbar:
        while cap.isOpened():
            # Read a frame
            ret, frame = cap.read()
            # If it's ok
            if ret:
                frame_number = frame_number + 1
                centers, bird_centers, frame, one_meter_threshold_bird, one_meter_threshold_yolo = \
                    detect_people_on_frame(
                        model,
                        frame,
                        confidence,
                        height,
                        width,
                        pts, frame_number, one_meter_threshold_bird, one_meter_threshold_yolo)
                '''
                print('frame n°', frame_number)
                print('#####centers####', centers)
                print('####bird_centers####', bird_centers)
                '''

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


def add_text(frame, height, width, shape, sdd_violations, yolo_violation):
    # Display yolov5 title
    cv2.putText(img=frame,
                text="Yolo v5",
                org=(int(width / 2 - 50), 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=2)

    # Display Bird Eye View title
    cv2.putText(img=frame,
                text="Bird Eye View",
                org=(int(width + width / 2 - 100), 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=2)

    # Display SDD (Social distancig detection) title
    cv2.putText(img=frame,
                text="SDD",
                org=(int(width * 2 + width / 2 - 40), 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=2)

    # Display the number of people in the first frame
    cv2.putText(img=frame,
                text="#People:" + str(shape[0]),
                org=(30, height + 85),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=2)

    # Display the number of violations in the first frame
    cv2.putText(img=frame,
                text="#Violations:" + str(yolo_violation),
                org=(220, height + 85),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255),
                thickness=2)

    # Display the number of people in the second and third frame
    cv2.putText(img=frame,
                text="#People:" + str(shape[0]),
                org=(width * 2 - 190, height + 85),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=2)

    # Display the number of violations in second and third the frame
    cv2.putText(img=frame,
                text="#Violations:" + str(sdd_violations),
                org=(width * 2, height + 85),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255),
                thickness=2)
