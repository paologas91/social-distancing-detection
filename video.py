import os
import cv2


def convert_video(path):
    """
    Converts the video in a compressed mp4 version
    :param path: The path of the video to convert
    :return:
    """
    # split a given input string into different substrings based on a delimiter and take the first cell of the array
    filename = os.path.basename(os.path.normpath(path))
    compressed_path = filename.split('.')[0]  # "test"
    print(compressed_path)
    compressed_path = 'compressed_' + compressed_path + '.mp4'  # "test.mp4"

    # remove the file if already exists
    if os.path.exists(compressed_path):
        os.remove(compressed_path)

    # Convert video in a specific format
    os.system(f"ffmpeg -i {path} -vcodec libx264 -vf scale=500:-2 {compressed_path}")
    return compressed_path


def display_video(filename):
    """
    Displays the video
    :param filename: The name of the file to play
    :return:
    """
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
            # Display the resulting frame
            cv2.imshow(filename, frame)

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


def get_video_properties(filename):
    """
    Gets the main properties of the video file
    :param filename: The video file
    :return: fps, width, height
    """
    cap = cv2.VideoCapture(filename)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, height, width


def recover_first_frame(filename):
    """
    Recovers the first frame of the selected video and saves it in the project path
    :param filename: The path of the video
    :return:
    """
    cap = cv2.VideoCapture(filename)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('first_frame.jpg', frame)
            break
    cap.release()


def saveVideo(fourcc, fps, width, height):
    """

    :return:
    """
    count = 0
    filename = "experiment" + str(count) + ".avi"
    folder = "./experiment/"
    path = folder + filename
    if not os.path.exists(folder):
        os.makedirs(folder)
    while os.path.exists(path):
        count = count + 1
        path = folder + "experiment" + str(count) + ".avi"
    print(filename)
    out = cv2.VideoWriter(path, fourcc, fps, (width * 2, height))
    return out
