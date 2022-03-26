from base64 import b64encode
#from google.colab import files
#from google.colab.patches import cv2_imshow
#from IPython.display import HTML
#from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import os
import torch


# input path: "test.mp4"
def convert_video(path):
    # split a given input string into different substrings based on a delimiter and take the first cell of the array
    compressed_path = path.split('.')[0]  # "test"
    compressed_path = 'compressed_' + compressed_path + '.mp4'  # "test.mp4"

    # remove the file if already exists
    if os.path.exists(compressed_path):
        os.remove(compressed_path)

    # Convert video in a specific format
    os.system(f"C:\\Users\\pgasp\\Desktop\\ffmpeg-5.0-essentials_build\\bin\\ffmpeg -i {path} -vcodec libx264 {compressed_path}")

    # Show video in html video player
    '''mp4 = open(compressed_path, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML("""
    <video width=800 controls>
        <source src="%s" type="video/mp4">
    </video>
    """ % data_url)'''


filename = 'campus4-c0.avi'
convert_video(filename)

# Load model

model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True, verbose=False)
model.cuda('cuda:0')


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


def detect_people_on_frame(img, confidence, distance):
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

    colors = ['green']*len(xyxy)
    for i in range(len(xyxy)):
        for j in range(i+1, len(xyxy)):
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


def detect_people_on_video(filename, confidence=0.9, distance=60):
    """Detect people on a video and draw the rectangles and lines."""
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
                frame = detect_people_on_frame(frame, confidence, distance)
                # Write new video
                out.write(frame)
                pbar.update(1)
            else:
                break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


detect_people_on_video(filename, confidence=0.5)
