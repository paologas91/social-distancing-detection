import sys
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from functions import *
from model import load_model
from video import *
from detect import bird_detect_people_on_video


# select and open video file
Tk().withdraw()
filename = askopenfilename(title='Select a video file...', filetypes=[("all video format", ".mp4"),
                                                                      ("all video format", ".flv"),
                                                                      ("all video format", ".avi")
                                                                      ])

if filename != "":
    # convert video
    convert_video(filename)

    # display video
    display_video(filename)

    # Load model
    model = load_model('x')

    # Get video properties
    fps, height, width = get_video_properties(filename)

    # recover the four points and ask to confirm the choice
    answer = False
    while not answer:
        pts = recover_four_points(filename, width)
        answer = ask_to_confirm()

    # compute the top-down perspective (bird's eye view)
    # compute_bird_eye()


    # detect people and compute distances among people
    # detect_people_on_video(model, filename, confidence=0.5)
    detect_people_on_video(model, filename, fps, height, width, pts, confidence=0.5)
else:
    sys.exit()

