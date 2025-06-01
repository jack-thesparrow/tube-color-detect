import cv2 as cv
import numpy as np
import os


# To load and preprocess the image so that it can be handled properly
def resize_to_target(img, target_w=1280, target_h=720):
    h, w = img.shape[:2]

    # Compute scale to fit target dimensions
    scale_w = target_w / w
    scale_h = target_h / h
    scale = min(scale_w, scale_h)  # Use min to ensure fit *within* the box

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
    print(f"Resized image to: {new_w}x{new_h}")
    return resized


def load_and_preprocess(img_path):
    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at path: {img_path}")

    img = resize_to_target(img, 1280, 720)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    canny = cv.Canny(blur, 180, 400)

    return img, hsv, canny


# To find the region of interest
def find_bottle_contour(eroded_img):
    contours, hierarchies = cv.findContours(
        eroded_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    bottle_contour = [cnt for cnt in contours if cv.boundingRect(cnt)[3] > 105]
    bottle_contour.sort(key=lambda cnt: cv.boundingRect(cnt)[0])
    return bottle_contour


def bottle_annotation(contours, img, output_path):
    i = 0
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        cv.putText(
            img,
            "tube" + str(i),
            (x, y - 5),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
        )
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        i = i + 1

    cv.imwrite(output_path, img)


def main(images_folder):
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(images_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            full_path = os.path.join(images_folder, filename)
            try:
                img, hsv, eroded = load_and_preprocess(full_path)
                contours = find_bottle_contour(eroded)
                output_filename = os.path.splitext(filename)[0] + "_annotated.png"
                output_path = os.path.join(output_folder, output_filename)
                bottle_annotation(contours, img, output_path)
                print(f"Processed and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    main("assets")
