import cv2 as cv
import numpy as np
import os

# === Setup ===
script_dir = os.path.dirname(os.path.abspath(__file__))
file = os.path.join(script_dir, "test_tube_empty_BGR.png")

img = cv.imread(file)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# === HSV Ranges ===
color_hsv_ranges = {
    "red1": ([0, 100, 100], [10, 255, 255]),
    "red2": ([160, 100, 100], [179, 255, 255]),
    "cyan": ([80, 100, 100], [100, 255, 255]),
    "green": ([45, 100, 100], [75, 255, 255]),
    "pink": ([140, 100, 150], [160, 255, 255]),
    "tan": ([10, 50, 150], [20, 180, 255]),
    "white": ([0, 0, 200], [179, 30, 255]),
    "orange": ([10, 150, 150], [25, 255, 255]),
    "lime": ([35, 100, 100], [45, 255, 255]),
    "rose": ([160, 50, 150], [170, 255, 255]),
    "blue": ([100, 100, 100], [130, 255, 255]),
    "lavender": ([125, 50, 150], [140, 255, 255]),
    "purple": ([140, 100, 100], [155, 255, 255]),
    "brown": ([10, 100, 20], [20, 200, 200]),
    "black": ([0, 0, 0], [180, 255, 50]),
}

bgr_colors = {
    "red": (0, 0, 255),
    "cyan": (255, 255, 0),
    "green": (0, 255, 0),
    "pink": (255, 0, 255),
    "tan": (180, 200, 140),
    "white": (255, 255, 255),
    "orange": (0, 165, 255),
    "lime": (0, 255, 150),
    "rose": (255, 102, 204),
    "blue": (255, 0, 0),
    "lavender": (230, 190, 255),
    "purple": (128, 0, 128),
    "brown": (42, 42, 165),
    "black": (50, 50, 50),
}

font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1

color_masks = {}
color_boxes = {}
non_black_areas = []

# Combine red1 and red2
red_mask = cv.bitwise_or(
    cv.inRange(
        hsv,
        np.array(color_hsv_ranges["red1"][0]),
        np.array(color_hsv_ranges["red1"][1]),
    ),
    cv.inRange(
        hsv,
        np.array(color_hsv_ranges["red2"][0]),
        np.array(color_hsv_ranges["red2"][1]),
    ),
)
color_masks["red"] = red_mask
color_boxes["red"] = []

# Process other colors
for color in color_hsv_ranges:
    if color in ["red1", "red2"]:
        continue
    lower, upper = np.array(color_hsv_ranges[color][0]), np.array(
        color_hsv_ranges[color][1]
    )
    mask = cv.inRange(hsv, lower, upper)
    color_masks[color] = mask
    color_boxes[color] = []

# Extract bounding boxes and compute areas
for color, mask in color_masks.items():
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        area = w * h
        if area > 100:  # Noise filter
            color_boxes[color].append((x, y, w, h))
            if color != "black":
                non_black_areas.append(area)

# Reference area from non-black
if not non_black_areas:
    print("No non-black regions found.")
    exit()

ref_area = min(non_black_areas)

# Draw boxes
for color, boxes in color_boxes.items():
    draw_color = bgr_colors.get(color, (200, 200, 200))
    for x, y, w, h in boxes:
        area = w * h

        if color == "black":
            if area >= ref_area / 2:
                cv.rectangle(img, (x, y), (x + w, y + h), draw_color, 2)
                cv.putText(
                    img,
                    "empty",
                    (x + 3, y + 15),
                    font,
                    font_scale,
                    draw_color,
                    font_thickness,
                    cv.LINE_AA,
                )
        else:
            if area < ref_area / 2:
                continue  # Skip small blobs
            elif area > 1.5 * ref_area:
                # Split into segments
                num_segments = int(round(h / (ref_area / w)))
                seg_height = h // num_segments
                for i in range(num_segments):
                    sy = y + i * seg_height
                    sh = seg_height if i < num_segments - 1 else (y + h) - sy
                    cv.rectangle(img, (x, sy), (x + w, sy + sh), draw_color, 2)
                    cv.putText(
                        img,
                        color,
                        (x + 3, sy + 15),
                        font,
                        font_scale,
                        draw_color,
                        font_thickness,
                        cv.LINE_AA,
                    )
            else:
                # Normal box
                cv.rectangle(img, (x, y), (x + w, y + h), draw_color, 2)
                cv.putText(
                    img,
                    color,
                    (x + 3, y + 15),
                    font,
                    font_scale,
                    draw_color,
                    font_thickness,
                    cv.LINE_AA,
                )

# Show result
cv.imshow("Tube Segment Detection", img)
cv.waitKey(0)
cv.destroyAllWindows()
