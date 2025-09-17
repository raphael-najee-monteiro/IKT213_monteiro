import cv2
import numpy as np

def sobel_edge_detection(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    sobel_image = cv2.Sobel(blur_image, ksize=1, dx=1, dy=1, ddepth=cv2.CV_64F)
    cv2.imwrite("solutions/sobel.jpg", sobel_image)

def canny_edge_detection(image, threshold1, threshold2):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    canny_image = cv2.Canny(blur_image, threshold1, threshold2)
    cv2.imwrite("solutions/canny.jpg", canny_image)

def template_match(image, template):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    w, h = template.shape[::-1]

    matched = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(matched >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)

    cv2.imwrite("solutions/matched_template.jpg", image)

def resize(image, scale_factor, up_or_down):
    result = image.copy()

    if up_or_down == "up":
        for _ in range(scale_factor):
            result = cv2.pyrUp(result)
    elif up_or_down == "down":
        for _ in range(scale_factor):
            result = cv2.pyrDown(result)
    else:
        raise ValueError("up_or_down must be 'up' or 'down'")

    cv2.imwrite("solutions/resize.jpg", result)


if __name__ == "__main__":
    img_lambo = cv2.imread("../resources/lambo.png")
    sobel_edge_detection(img_lambo)
    canny_edge_detection(img_lambo, 50, 50)
    
    img = cv2.imread("../resources/shapes.png")
    templ = cv2.imread("../resources/shapes_template.jpg", 0)
    template_match(img, templ)

    img_resize = cv2.imread("../resources/lambo.png")
    resize(img_resize, 2, "down")