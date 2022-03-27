# from google.colab import files
# from google.colab.patches import cv2_imshow
# from IPython.display import HTML
# from PIL import Image

from functions import *

# convert video
filename = 'campus4-c0.avi'
convert_video(filename)

# display video
# display_video('compressed_campus4-c0.mp4')

# Load model
model = load_model()

# detect people
# detect_people_on_video(model, filename, confidence=0.5)

recover_four_points(filename)
