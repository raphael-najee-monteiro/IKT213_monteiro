import cv2
import numpy as np


# 1) padding
def padding(image, border_width):
    padded_image = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)
    cv2.imwrite("solutions/lena_padded.png", padded_image)

# 2) cropping
def crop(image, x_0, x_1, y_0, y_1):
    cropped_image = image[y_0:y_1, x_0:x_1]
    cv2.imwrite("solutions/lena_cropped.png", cropped_image)

# 3) resize
def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    cv2.imwrite("solutions/lena_resized.png", resized_image)

# 4) copy
def copy(image, emptyPictureArray):
    height, width, channels = image.shape

    for y in range(height):
        for x in range(width):
            emptyPictureArray[y, x] = image[y, x]

    cv2.imwrite("solutions/lena_copy.png", emptyPictureArray)

# 5) grayscale
def grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("solutions/lena_grayscale.png", grayscale_image)

# 6) HSV
def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite("solutions/lena_hsv.png", hsv_image)

# 7) color shift
def hue_shifted(image, hue):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    h = ((h.astype(np.int16) + hue) % 180).astype(np.uint8)
    merged_image = cv2.merge([h, s, v])
    cv2.imwrite("solutions/lena_hue_shifted.png", merged_image)

# 8) smoothing
def smoothing(image):
    smoothed_image = cv2.GaussianBlur(image, (15, 15), cv2.BORDER_DEFAULT)
    cv2.imwrite("solutions/lena_smoothed.png", smoothed_image)

# 9) rotation
def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    else:
        rotated_image = image

    cv2.imwrite("solutions/lena_rotated.png", rotated_image)



if __name__ == "__main__":
    img = cv2.imread("../resources/lena.png")
    h, w, ch = img.shape
    empty = np.zeros((h, w, ch), dtype=np.uint8)

    padding(img, 100)
    crop(img, 80, w - 130, 80, h - 130)
    resize(img, 200, 200)
    copy(img, empty)
    grayscale(img)
    hsv(img)
    hue_shifted(img, 50)
    smoothing(img)
    rotation(img, 90)