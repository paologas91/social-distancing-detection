import sys
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from distance import get_distance_from_video
from functions import *
from model import load_model
from video import *
from image import *
from detect import detect_people_on_video


# select and open video file
Tk().withdraw()
filename = askopenfilename(title='Select a video file...', filetypes=[("all video format", ".mp4"),
                                                                      ("all video format", ".flv"),
                                                                      ("all video format", ".avi")
                                                                      ])

if filename != "":





    # convert video
    filename_compressed = convert_video(filename)

    # display video
    #display_video(filename_compressed)

    # Load model
    model = load_model('x')

    # Get video properties
    fps, height, width = get_video_properties(filename_compressed)

    # Recover the first frame of the selected video
    recover_first_frame(filename_compressed)

    # Draw two black stripes at the left and at the right of the first frame
    #draw_img_with_black_stripes('first_frame.jpg', width)

    # Recover the four points and ask to confirm the choice
    answer = False
    while not answer:
        pts = recover_four_points()
        print("pts: \n", pts)
        windowName='first_frame_with_polygon.jpg'
        answer = ask_to_confirm(windowName)

    # compute the top-down perspective (bird's eye view)
    # compute_bird_eye()


    # take distance
    #distance = get_distance_from_video(filename)
    #print("distance:", distance)

    # detect people and compute distances among people
    #detect_people_on_video2(model, filename, confidence=0.5)
    distance=(input("insert de number of metro of distance:"))
    distance=int(distance)*100
    detect_people_on_video(model, filename_compressed, fps, height, width, pts, distance,confidence=0.5)
else:
    sys.exit()

