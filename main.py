import cv2 as cv
import numpy as np


# To load and preprocess the image so that it can be handled properly
def load_and_preprocess(img_path, scale=0.65):
    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {img_path}")

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


# To find the region of interest
def find_bottle_contour(edged_img):
    contours, hierarchies = cv.findContours(
        edged_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )  # to find only the external edges of the objects
    bottle_contour = [
        cnt for cnt in contours if cv.boundingRect(cnt)[3] > 100
    ]  # added list comprehension which basically says remove bounded small objects from contours list with height less than 100px
    bottle_contour.sort(
        key=lambda cnt: cv.boundingRect(cnt)[0]
    )  # This sorts the x coordinates found in order from left to right(ascending)
    return bottle_contour


def bottle_anotation(contours, img):
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imwrite("annotated_result.png", img)


def main(img_path):
    img, hsv, edged = load_and_preprocess(img_path)
    # cv.imshow("Edges", edged)
    # cv.waitKey(0)
    bottle_contour = find_bottle_contour(edged)
    bottle_anotation(bottle_contour, img)


if __name__ == "__main__":
    main(r"assets/bottles.jpg")
