import cv2


def draw_img_with_black_stripes(filename, width):
    img = cv2.imread(filename)
    color = (0, 0, 0)
    left, right = [int(width / 2)] * 2
    img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=color)
    cv2.imwrite('first_frame_with_black_stripes.jpg', img)
