import cv2 as cv
import numpy as np
import json
import os


def scale_hsv(h, s, v):
    """Convert HSV from standard (0–360, 0–100, 0–100) to OpenCV (0–179, 0–255, 0–255)."""
    h = int((h % 360) / 2)
    s = int((s / 100) * 255)
    v = int((v / 100) * 255)
    return [h, s, v]


def get_strict_mask(hsv_img, color, strict_factor=1):
    """
    Generate a strict mask for the given HSV color.
    `strict_factor` lowers the range to reduce false positives.
    Handles red hue wrapping around 0/179.
    """
    h, s, v = color
    h_range = int(10 * strict_factor)
    s_range = int(50 * strict_factor)
    v_range = int(50 * strict_factor)

    lower_s = max(s - s_range, 0)
    upper_s = min(s + s_range, 255)
    lower_v = max(v - v_range, 0)
    upper_v = min(v + v_range, 255)

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


def auto_segment_and_annotate(
    img, known_colors, strict_factor=0.6, min_segment_height=20, pixel_threshold=300
):
    """
    Auto-detect color segments vertically by looking for color changes,
    annotate each segment with detected dominant color,
    and return the annotated image plus a JSON-compatible summary.
    """
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]

    # Detect dominant color per horizontal row
    row_colors = []
    for y in range(h):
        row = hsv[y, :, :]
        color_scores = {}
        for name, hsv_val in known_colors.items():
            mask = get_strict_mask(row.reshape(1, w, 3), hsv_val, strict_factor)
            color_scores[name] = cv.countNonZero(mask)
        best_color = max(color_scores, key=color_scores.get)
        # If count below threshold, treat as "unknown"
        if color_scores[best_color] < pixel_threshold:
            best_color = "unknown"
        row_colors.append(best_color)

    # Segment rows by consecutive color changes
    segments = []
    start = 0
    for i in range(1, len(row_colors)):
        if row_colors[i] != row_colors[i - 1]:
            if row_colors[i - 1] != "unknown" and (i - start) >= min_segment_height:
                segments.append((start, i - 1, row_colors[i - 1]))
            start = i
    # Last segment
    if row_colors[-1] != "unknown" and (len(row_colors) - start) >= min_segment_height:
        segments.append((start, len(row_colors) - 1, row_colors[-1]))

    annotated = img.copy()

    # Annotate each segment
    for y_start, y_end, color_name in segments:
        center_y = (y_start + y_end) // 2
        cv.putText(
            annotated,
            color_name,
            (10, center_y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv.LINE_AA,
        )
        # Optional: draw horizontal line to mark segment boundaries
        cv.line(annotated, (0, y_start), (w, y_start), (0, 255, 0), 1)
        cv.line(annotated, (0, y_end), (w, y_end), (0, 255, 0), 1)

    # Prepare JSON output: list of segments with color and pixel ranges
    json_summary = []
    for idx, (y_start, y_end, color_name) in enumerate(segments):
        json_summary.append(
            {
                "segment": idx,
                "start_pixel": y_start,
                "end_pixel": y_end,
                "color": color_name,
            }
        )

    return annotated, json_summary


if __name__ == "__main__":
    # === Configuration ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_image_path = os.path.join(script_dir, "../tubes/bottle0_tube_strip5.png")
    output_image_path = "annotated_output.png"
    output_json_path = "color_summary.json"

    # Define your known colors in HSV (OpenCV format)
    known_colors = {
        "red": scale_hsv(359.2, 62.6, 89.0),
        "tan": scale_hsv(32.7, 35.6, 87.1),
        "rose": scale_hsv(359.3, 35.3, 98.8),
        "purple": scale_hsv(257.9, 57.9, 79.2),
        "blue": scale_hsv(219.9, 76.2, 98.0),
        "white": scale_hsv(180.0, 3.7, 94.1),
        "orange": scale_hsv(32.2, 87.7, 99.2),
        "cyan": scale_hsv(189.5, 85.5, 81.2),
        "lime": scale_hsv(119.1, 63.4, 79.2),
    }

    img = cv.imread(input_image_path)
    if img is None:
        print(f"Failed to load image: {input_image_path}")
        exit(1)

    annotated_img, color_json = auto_segment_and_annotate(img, known_colors)

    # Save outputs
    cv.imwrite(output_image_path, annotated_img)
    with open(output_json_path, "w") as f:
        json.dump(color_json, f, indent=4)

    print(f"Annotated image saved to {output_image_path}")
    print(f"Color JSON summary saved to {output_json_path}")

    # Optionally show image
    cv.imshow("Annotated Image", annotated_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
