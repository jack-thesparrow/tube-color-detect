import cv2 as cv
import numpy as np
import os


# Crop the image with adjustable start and size (non-square support)
def crop_img(img, crop_w=None, crop_h=None, x_start=None, y_start=None):
    height, width = img.shape[:2]
    print(f"Original image size: {width}px x {height}px")

    if crop_w is None:
        crop_w = width
    if crop_h is None:
        crop_h = height
    if x_start is None:
        x_start = 0
    if y_start is None:
        y_start = 0

    # Clamp so we don't exceed image bounds
    x_start = max(0, min(x_start, width - crop_w))
    y_start = max(0, min(y_start, height - crop_h))

    crop = img[y_start : y_start + crop_h, x_start : x_start + crop_w]
    print(f"Cropped image shape: {crop.shape}")
    return crop


# Load and preprocess image (crop, grayscale, HSV, edge detection)
def load_and_preprocess(img_path, crop_w=None, crop_h=None, x_start=None, y_start=None):
    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at path: {img_path}")

    img = crop_img(img, crop_w=crop_w, crop_h=crop_h, x_start=x_start, y_start=y_start)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    canny = cv.Canny(blur, 180, 400)

    return img, hsv, canny


# Find contours that likely represent bottles/tubes
def find_bottle_contour(eroded_img):
    contours, _ = cv.findContours(eroded_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Filter small objects and sort left-to-right
    bottle_contour = [cnt for cnt in contours if cv.boundingRect(cnt)[3] > 40]
    bottle_contour.sort(key=lambda cnt: cv.boundingRect(cnt)[0])
    return bottle_contour


# Annotate and save individual tubes and the annotated image
def bottle_annotation(contours, img, output_path, tubes_out_path, base_name):
    for i, cnt in enumerate(contours):
        x, y, w, h = cv.boundingRect(cnt)

        # Save individual tube
        tube = img[y : y + h, x : x + w]
        tube_path = os.path.join(tubes_out_path, f"{base_name}_tube{i}.png")
        cv.imwrite(tube_path, tube)

        # Draw label and rectangle
        cv.putText(
            img, f"tube{i}", (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1
        )
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the annotated image
    cv.imwrite(output_path, img)


def main(images_folder):
    output_folder = "output"
    tubes_out_folder = "tubes"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(tubes_out_folder, exist_ok=True)

    # Cropping parameters
    x_start = 330
    y_start = 240
    crop_w = 540
    crop_h = 780

    for filename in os.listdir(images_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            full_path = os.path.join(images_folder, filename)
            try:
                img, hsv, eroded = load_and_preprocess(
                    full_path,
                    crop_w=crop_w,
                    crop_h=crop_h,
                    x_start=x_start,
                    y_start=y_start,
                )
                contours = find_bottle_contour(eroded)
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, f"{base_name}_annotated.png")
                bottle_annotation(
                    contours, img, output_path, tubes_out_folder, base_name
                )
                print(f"Processed and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    main("assets")
