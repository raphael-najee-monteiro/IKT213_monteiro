import cv2


def print_image_information(image):
    height, width, channels = image.shape

    print("Height:", height)
    print("Width:", width)
    print("Channels:", channels)
    print("Size (number of values):", image.size)
    print("Data type:", image.dtype)


def print_cam_information():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    fps = cam.get(cv2.CAP_PROP_FPS)
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

    with open("solutions/camera_outputs.txt", "w") as f:
        f.write("FPS:" + str(fps) + "\n")
        f.write("Width:" + str(width) + "\n")
        f.write("Height:" + str(height) + "\n")

    cam.release()

img = cv2.imread("lena-1.png")
print_image_information(img)

print_cam_information()