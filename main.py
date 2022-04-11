from tkinter import Tk
from tkinter.filedialog import askopenfilename
from distance import *
from roi import *
from model import *
from video import *
from detect import *


mouse_pts = []

# Select and open a video file
Tk().withdraw()
filename = askopenfilename(title='Select a video file...', filetypes=[("all video format", ".mp4"),
                                                                      ("all video format", ".flv"),
                                                                      ("all video format", ".avi")
                                                                      ])

if filename != "":

    # Convert video
    filename_compressed = convert_video(filename)

    # Display video
    display_video(filename_compressed)

    # Load model
    model = load_model('s')

    # Get video properties
    fps, height, width = get_video_properties(filename_compressed)

    # Recover the first frame of the selected video
    recover_first_frame(filename_compressed)

    # Recover the ROI and ask to confirm the choice
    answer = False
    while not answer:
        mouse_pts = recover_roi_points()
        window_name = 'first_frame_with_polygon.jpg'
        answer = ask_to_confirm_roi(window_name)

    # Recover the two points of the distance and ask to confirm the choice
    answer = False
    while not answer:
        choose_distance_frame(filename_compressed)
        print("Draw a distance line by selecting two points on the frame.")
        distance_pts = recover_distance_points()
        insert_distance_in_meters()
        window_name = 'distance_frame_with_line.jpg'
        answer = ask_to_confirm_distance(window_name)

    # Run the inference and shows the results
    detect_people_on_video(model, filename_compressed, fps, height, width, mouse_pts, confidence=0.5)

else:
    sys.exit()
