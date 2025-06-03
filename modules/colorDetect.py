import cv2 as cv
import numpy as np
import os
from collections import defaultdict

# === Configuration ===
script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(script_dir, "../tubes")
output_folder = os.path.join(script_dir, "../color")

# HSV color ranges (adjust if needed)
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

# BGR colors for annotation
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


def extract_color_boxes(hsv_img):
    """Detect color masks and extract bounding boxes per color."""
    color_masks = {}
    color_boxes = defaultdict(list)

    # Combine red ranges
    red_mask = cv.bitwise_or(
        cv.inRange(
            hsv_img,
            np.array(color_hsv_ranges["red1"][0]),
            np.array(color_hsv_ranges["red1"][1]),
        ),
        cv.inRange(
            hsv_img,
            np.array(color_hsv_ranges["red2"][0]),
            np.array(color_hsv_ranges["red2"][1]),
        ),
    )
    color_masks["red"] = red_mask

    # Other colors
    for color in color_hsv_ranges:
        if color in ["red1", "red2"]:
            continue
        lower, upper = np.array(color_hsv_ranges[color][0]), np.array(
            color_hsv_ranges[color][1]
        )
        color_masks[color] = cv.inRange(hsv_img, lower, upper)

    # Find contours and bounding boxes
    for color, mask in color_masks.items():
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            area = w * h
            color_boxes[color].append((x, y, w, h, area))

    return color_boxes


def main():
    # Load all image paths
    image_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not image_files:
        print("No images found in folder:", input_folder)
        return

    # Step 1: Calculate reference area from all tubes (using middle strips)
    ref_area = None
    for file in image_files:
        img = cv.imread(file)
        strip = img[3:-3, 17:-17]  # crop middle strip
        hsv_strip = cv.cvtColor(strip, cv.COLOR_BGR2HSV)
        boxes = extract_color_boxes(hsv_strip)
        for color, rects in boxes.items():
            if color == "black":
                continue
            for _, _, _, _, area in rects:
                if ref_area is None or area < ref_area:
                    ref_area = area

    if ref_area is None:
        print("No non-black color areas found across images.")
        return

    print(f"Reference color area: {ref_area}")

    # Step 2: Process each tube for color detection and annotation
    for idx, file in enumerate(image_files):
        img = cv.imread(file)
        original = img.copy()
        strip = img[3:-3, 15:-15]
        hsv_strip = cv.cvtColor(strip, cv.COLOR_BGR2HSV)

        color_boxes = extract_color_boxes(hsv_strip)
        tube_colors = []

        # Detect and record colors above half reference area
        for color, boxes in color_boxes.items():
            for x, y, w, h, area in sorted(
                boxes, key=lambda b: b[1]
            ):  # sort by vertical position
                if area < ref_area / 2:
                    continue
                tube_colors.append(color)
                # Draw box on original image (adjusting coords for strip offset)
                cv.rectangle(
                    original,
                    (x + 15, y + 3),
                    (x + w + 15, y + h + 3),
                    bgr_colors.get(color, (200, 200, 200)),
                    2,
                )
                cv.putText(
                    original,
                    color,
                    (x + 18, y + 15),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    bgr_colors.get(color, (200, 200, 200)),
                    1,
                    cv.LINE_AA,
                )

        # Estimate empty capacity (how many segments black areas can hold)
        black_boxes = color_boxes.get("black", [])
        empty_capacity = 0
        for _, _, _, h, area in black_boxes:
            if area >= ref_area / 2:
                empty_capacity += int(
                    round(h / (ref_area / (h if h > 0 else 1)))
                )  # fallback to h to avoid div by zero

        # Print output for the tube
        print(f"Tube {idx + 1}:")
        print("  Colors:", tube_colors if tube_colors else ["empty"])
        print("  Empty capacity:", empty_capacity)

        # Save or show annotated image
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            cv.imwrite(
                os.path.join(output_folder, f"annotated_tube_{idx+1}.png"), original
            )
        else:
            cv.imshow(f"Annotated Tube {idx+1}", original)
            cv.waitKey(0)
            cv.destroyAllWindows()


if __name__ == "__main__":
    main()
