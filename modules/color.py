import cv2 as cv
import numpy as np
import os


# === Scale HSV from human-friendly to OpenCV format ===
def scale_hsv(h, s, v):
    h = int((h % 360) / 2)  # OpenCV hue: 0–179
    s = int((s / 100) * 255)
    v = int((v / 100) * 255)
    return [h, s, v]


# === Your Known Colors with updated purple ===
known_colors = {
    "red": scale_hsv(359.2, 62.6, 89.0),
    "tan": scale_hsv(32.7, 35.6, 87.1),
    "rose": scale_hsv(359.3, 35.3, 98.8),
    # Purple updated here with average saturation and value around your ranges
    "purple": [138, 183, 199],
    "blue": scale_hsv(219.9, 76.2, 98.0),
    "white": scale_hsv(180.0, 3.7, 94.1),
    "orange": scale_hsv(32.2, 87.7, 99.2),
    "cyan": scale_hsv(189.5, 85.5, 81.2),
    "lime": scale_hsv(119.1, 63.4, 79.2),
}

# === Per-Color Tolerances (H, S, V) with updated purple ===
tolerances = {
    "red": (5, 40, 40),
    "tan": (5, 40, 40),
    "rose": (5, 40, 40),
    "purple": (4, 40, 40),  # Hue ±4, Sat ±40, Val ±40
    "blue": (5, 40, 40),
    "white": (5, 30, 30),
    "orange": (5, 40, 40),
    "cyan": (5, 40, 40),
    "lime": (5, 40, 40),
}


def get_strict_mask(hsv_img, color, tol):
    h, s, v = color
    h_tol, s_tol, v_tol = tol

    lower_s = max(s - s_tol, 0)
    upper_s = min(s + s_tol, 255)
    lower_v = max(v - v_tol, 0)
    upper_v = min(v + v_tol, 255)

    h_range = h_tol
    if h - h_range < 0:
        lower1 = np.array([0, lower_s, lower_v], dtype=np.uint8)
        upper1 = np.array([h + h_range, upper_s, upper_v], dtype=np.uint8)
        lower2 = np.array([180 + (h - h_range), lower_s, lower_v], dtype=np.uint8)
        upper2 = np.array([179, upper_s, upper_v], dtype=np.uint8)

        mask1 = cv.inRange(hsv_img, lower1, upper1)
        mask2 = cv.inRange(hsv_img, lower2, upper2)
        return cv.bitwise_or(mask1, mask2)

    elif h + h_range > 179:
        lower1 = np.array([h - h_range, lower_s, lower_v], dtype=np.uint8)
        upper1 = np.array([179, upper_s, upper_v], dtype=np.uint8)
        lower2 = np.array([0, lower_s, lower_v], dtype=np.uint8)
        upper2 = np.array([(h + h_range) % 180, upper_s, upper_v], dtype=np.uint8)

        mask1 = cv.inRange(hsv_img, lower1, upper1)
        mask2 = cv.inRange(hsv_img, lower2, upper2)
        return cv.bitwise_or(mask1, mask2)

    else:
        lower = np.array([h - h_range, lower_s, lower_v], dtype=np.uint8)
        upper = np.array([h + h_range, upper_s, upper_v], dtype=np.uint8)
        return cv.inRange(hsv_img, lower, upper)


def largest_vertical_region_height(mask):
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
        mask, connectivity=8
    )
    max_height = 0
    for i in range(1, num_labels):  # skip background label 0
        x, y, w, h, area = stats[i]
        if h > max_height:
            max_height = h
    return max_height


def annotate_colors_vertically(
    img, known_colors, tolerances, num_bands=6, min_color_region_height=80
):
    h, w = img.shape[:2]
    band_height = h // num_bands
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    annotated = img.copy()

    band_colors = []
    band_confidences = []

    for i in range(num_bands):
        y_start = i * band_height
        y_end = (i + 1) * band_height if i != num_bands - 1 else h
        band = hsv[y_start:y_end, :]

        color_scores = {}
        for name, hsv_val in known_colors.items():
            mask = get_strict_mask(band, hsv_val, tolerances[name])

            max_region_height = largest_vertical_region_height(mask)

            if max_region_height >= min_color_region_height:
                color_scores[name] = cv.countNonZero(mask)
            else:
                color_scores[name] = 0

        best_color = max(color_scores, key=color_scores.get)
        confidence = color_scores[best_color]

        if confidence < 200:
            best_color = "unknown"

        band_colors.append(best_color)
        band_confidences.append(confidence)

    # Merge consecutive bands with the same color
    merged = []
    start = 0
    for i in range(1, num_bands + 1):
        if i == num_bands or band_colors[i] != band_colors[start]:
            merged.append((start, i - 1, band_colors[start]))
            start = i

    # Print and annotate merged bands
    for start_idx, end_idx, color_name in merged:
        y_start = start_idx * band_height
        y_end = (end_idx + 1) * band_height if end_idx != num_bands - 1 else h
        center_y = (y_start + y_end) // 2

        print(f"Bands {start_idx}–{end_idx}: {color_name}")

        center_x = w // 2
        cv.putText(
            annotated,
            color_name,
            (center_x - 30, center_y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv.LINE_AA,
        )

    return annotated


# === Run on Uploaded Image ===
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(
        script_dir, "../tubes/bottle0_tube_strip4.png"
    )  # Update path as needed
    img = cv.imread(img_path)
    assert img is not None, "Image not loaded"

    annotated_img = annotate_colors_vertically(
        img, known_colors, tolerances, num_bands=6, min_color_region_height=20
    )

    cv.imshow("Annotated", annotated_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
