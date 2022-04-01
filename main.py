# from google.colab import files
# from google.colab.patches import cv2_imshow
# from IPython.display import HTML
# from PIL import Image
import numpy as np

from functions import *

# convert video
filename = '4p-c1.avi'
convert_video(filename)

# display video
display_video(filename)

# Load model
model = load_model()

# recover the four points and ask to confirm the choice
answer = False
while not answer:
    recover_four_points(filename)
    answer = ask_to_confirm()

# compute the top-down perspective (bird's eye view)
# compute_bird_eye()


# detect people and compute distances among people
# detect_people_on_video(model, filename, confidence=0.5)
bird_detect_people_on_video(model, filename, confidence=0.5)
