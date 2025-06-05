import cv2 as cv
import numpy as np
import os
import json


# === Convert HSV values to OpenCV scale ===
def scale_hsv(h, s, v):
    h = int((h % 360) / 2)  # Hue: 0–179 in OpenCV
    s = int((s / 100) * 255)
    v = int((v / 100) * 255)
    return [h, s, v]


# === Known Colors and Tolerances ===
known_colors = {
    "red": scale_hsv(359.2, 62.6, 89.0),
    "tan": scale_hsv(32.7, 35.6, 87.1),
    "rose": scale_hsv(359.3, 35.3, 98.8),
    "purple": [138, 183, 199],
    "blue": scale_hsv(219.9, 76.2, 98.0),
    "white": scale_hsv(180.0, 3.7, 94.1),
    "orange": scale_hsv(32.2, 87.7, 99.2),
    "cyan": scale_hsv(189.5, 85.5, 81.2),
    "lime": scale_hsv(119.1, 63.4, 79.2),
    "empty": scale_hsv(226, 37, 21),
}
tolerances = {
    "red": (5, 40, 40),
    "tan": (5, 40, 40),
    "rose": (5, 40, 40),
    "purple": (4, 40, 40),
    "blue": (5, 40, 40),
    "white": (5, 30, 30),
    "orange": (5, 40, 40),
    "cyan": (5, 40, 40),
    "lime": (5, 40, 40),
    "black": (5, 50, 40),
    "empty": (7, 30, 30),
}

# === Parameters ===
MIN_ROW_PIXEL_COUNT = 20
MIN_SEGMENT_HEIGHT = 30
ROW_STEP = 4


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


def get_color_segments(hsv_img, known_colors, tolerances, row_step=ROW_STEP):
    height = hsv_img.shape[0]
    last_color = None
    segments = []
    start_row = 0

    for row in range(0, height, row_step):
        row_slice = hsv_img[row : row + row_step, :]
        row_colors = {}

        for color_name, hsv_val in known_colors.items():
            mask = get_strict_mask(row_slice, hsv_val, tolerances[color_name])
            count = cv.countNonZero(mask)
            row_colors[color_name] = count if count >= MIN_ROW_PIXEL_COUNT else 0

        best_color = max(row_colors, key=row_colors.get)
        if row_colors[best_color] == 0:
            best_color = "black"

        if best_color != last_color:
            if last_color is not None and last_color != "black":
                segments.append((last_color, start_row, row))
            start_row = row
            last_color = best_color

    if last_color is not None and last_color != "black":
        segments.append((last_color, start_row, height))
    return segments


def merge_consecutive_segments(segments):
    merged = []
    for color, start, end in segments:
        if merged and merged[-1][0] == color:
            merged[-1] = (color, merged[-1][1], end)
        else:
            merged.append((color, start, end))
    return merged


def filter_short_segments(segments, min_height=MIN_SEGMENT_HEIGHT):
    return [(c, s, e) for (c, s, e) in segments if (e - s) >= min_height]


def get_segment_areas(hsv_img, segments, known_colors, tolerances):
    segment_areas = {}
    band_counts = {}
    for color_name, start, end in segments:
        band_counts[color_name] = band_counts.get(color_name, 0) + 1
        key = f"{color_name}_{band_counts[color_name]}"
        segment_slice = hsv_img[start:end, :]
        mask = get_strict_mask(
            segment_slice, known_colors[color_name], tolerances[color_name]
        )
        area = int(cv.countNonZero(mask))
        segment_areas[key] = area
    return segment_areas


def approximate_wide_bands(results, global_min):
    adjusted_results = {}
    for filename, data in results.items():
        new_order = []
        new_areas = {}
        for color_key, area in data["areas"].items():
            color = color_key.rsplit("_", 1)[0]
            ratio = area / global_min if global_min else 1
            estimated_slots = max(1, int(ratio))
            for _ in range(estimated_slots):
                new_order.append(color)
                new_areas[f"{color}_{len(new_areas) + 1}"] = area // estimated_slots
        adjusted_results[filename] = {
            "order": new_order,
            "areas": new_areas,
        }
    return adjusted_results


# === Main ===
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tubes_dir = os.path.join(script_dir, "../tubes")
    results = {}
    all_segment_areas = []

    for filename in sorted(os.listdir(tubes_dir)):
        if filename.lower().endswith(".png"):
            img_path = os.path.join(tubes_dir, filename)
            img = cv.imread(img_path)
            if img is None:
                print(f"Failed to load image: {filename}")
                continue

            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            segments = get_color_segments(hsv, known_colors, tolerances)
            segments = merge_consecutive_segments(segments)
            segments = filter_short_segments(segments)

            color_order = [seg[0] for seg in segments]
            color_areas = get_segment_areas(hsv, segments, known_colors, tolerances)

            results[filename] = {"order": color_order, "areas": color_areas}
            all_segment_areas.extend(color_areas.values())

    global_min_area = min(all_segment_areas) if all_segment_areas else None
    print(f"\nGlobal least area from all segments: {global_min_area}")

    adjusted_results = approximate_wide_bands(results, global_min_area)

    for filename, data in adjusted_results.items():
        print(f"\n{filename}:\n  Order: {data['order']}\n  Areas: {data['areas']}")

    output_path = os.path.join(script_dir, "tube_analysis.json")
    with open(output_path, "w") as f:
        json.dump(adjusted_results, f, indent=2)

    print(f"\nSaved analysis to: {output_path}")
