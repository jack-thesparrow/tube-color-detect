import cv2 as cv
import numpy as np
import os


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


def bottle_annotation(contours, img, output_path):
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imwrite(output_path, img)


def main(images_folder):
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(images_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            full_path = os.path.join(images_folder, filename)
            try:
                img, hsv, edged = load_and_preprocess(full_path)
                contours = find_bottle_contour(edged)
                output_filename = os.path.splitext(filename)[0] + "_annotated.png"
                output_path = os.path.join(output_folder, output_filename)
                bottle_annotation(contours, img, output_path)
                print(f"Processed and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    main("assets")
