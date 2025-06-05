import cv2 as cv


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

    # So that resolution remains same to get desired crop(as screenshot may be from different devices)
    img = cv.resize(img, (540, 1200))
    if img is None:
        raise FileNotFoundError(f"Could not load image at path: {img_path}")

    img = crop_img(img, crop_w=crop_w, crop_h=crop_h, x_start=x_start, y_start=y_start)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    canny = cv.Canny(blur, 180, 400)

    return img, hsv, canny
