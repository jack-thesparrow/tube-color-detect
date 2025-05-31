import cv2 as cv
import numpy as np


def load_and_preprocess(image_path, scale=0.65):
    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    # Rescaling the image cause its too big on my screen
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dimensions = (width, height)
    resized_img = cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
    gray = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(resized_img, cv.COLOR_BGR2HSV)
    blurred = cv.GaussianBlur(resized_img, (3, 3), cv.BORDER_DEFAULT)
    edged = cv.Canny(blurred, 150, 150)

    return resized_img, hsv, edged


img_path = r"assets/bottles.jpg"

img, hsv, edged = load_and_preprocess(img_path)
# Uncomment to see the loaded and transformed images
# cv.imshow("Image", img)
# cv.imshow("hsv", hsv)
# cv.imshow("edged", edged)

cv.waitKey(0)
