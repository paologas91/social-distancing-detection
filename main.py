import sys
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from distance import *
from functions import recover_four_points, ask_to_confirm
from model import *
from video import *
from image import *
from detect import *
global mouse_pts, distance_pts

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
    display_video(filename_compressed)

    # Load model
    model = load_model('s')

    # Get video properties
    fps, height, width = get_video_properties(filename_compressed)

    # Recover the first frame of the selected video
    recover_first_frame(filename_compressed)

    # Draw two black stripes at the left and at the right of the first frame
    # draw_img_with_black_stripes('first_frame.jpg', width)

    # Recover two points and ask to confirm the choice
    # Recover the ROI and ask to confirm the choice
    answer = False
    while not answer:
        mouse_pts = recover_four_points()
        window_name = 'first_frame_with_polygon.jpg'
        answer = ask_to_confirm(window_name)

    answer = False
    while not answer:
        choose_frame_to_draw_distance(filename_compressed)
        print("Draw a distance line by selecting two points on the frame.")
        distance_pts = recover_two_points()
        insert_distance_in_meters()
        window_name = 'train_frame_with_line.jpg'
        answer = ask_to_confirm(window_name)


    # compute the top-down perspective (bird's eye view)
    # compute_bird_eye()

    # detect people and compute distances among people
    # detect_people_on_video2(model, filename, confidence=0.5)
    # distance=(input("insert the distance in metro:"))
    # distance=int(distance)*100

    detect_people_on_video(model, filename_compressed, fps, height, width, mouse_pts, confidence=0.5)

else:
    sys.exit()
