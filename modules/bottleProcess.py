import cv2 as cv
from sklearn.cluster import KMeans
import numpy as np
import os
from modules.tube_analyzer import *

# === Your Known Colors with updated purple ===
known_colors = {
    "red": scale_hsv(359.2, 62.6, 89.0),
    "tan": scale_hsv(32.7, 35.6, 87.1),
    "rose": scale_hsv(359.3, 35.3, 98.8),
    "purple": [138, 183, 199],  # manually tuned values for purple
    "blue": scale_hsv(219.9, 76.2, 98.0),
    "white": scale_hsv(180.0, 3.7, 94.1),
    "orange": scale_hsv(32.2, 87.7, 99.2),
    "cyan": scale_hsv(189.5, 85.5, 81.2),
    "lime": scale_hsv(119.1, 63.4, 79.2),
}

# === Per-Color Tolerances (H, S, V) ===
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


def segment_image(img: np.ndarray, n_clusters: int = 7) -> np.ndarray:
    """
    Applies KMeans color segmentation to an image and returns the segmented image.

    Parameters:
        img (np.ndarray): Input image in BGR format (as read by cv.imread).
        n_clusters (int): Number of color clusters to use.

    Returns:
        np.ndarray: Segmented image in BGR format.
    """
    # Convert to RGB for clustering
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_rgb = cv.GaussianBlur(img_rgb, (5, 5), 0)

    # Reshape for clustering
    X = img_rgb.reshape(-1, 3)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(X)

    # Replace each pixel with its cluster centroid
    segmented = kmeans.cluster_centers_[kmeans.labels_]
    segmented = segmented.reshape(img_rgb.shape).astype("uint8")

    # Convert back to BGR for output
    segmented_bgr = cv.cvtColor(segmented, cv.COLOR_RGB2BGR)
    return segmented_bgr


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

        # Save individual tube and edges eroded to get most pure colors possible(basically added padding)
        tube = img[y + 3 : y + h - 3, x + 3 : x + w - 3]
        tube_strip = tube[25:-3, 15:-8]
        ori_height = tube_strip.shape[0]
        tube_strip = cv.resize(tube_strip, (30, ori_height))
        tube_strip = segment_image(tube_strip)
        # tube_path = os.path.join(tubes_out_path, f"{base_name}_tube{i}.png")
        tube_strip_path = os.path.join(tubes_out_path, f"{base_name}_tube_strip{i}.png")
        # tube_strip_path = os.path.join(tubes_out_path, f"tube_strip{i}.png")

        # cv.imwrite(tube_path, tube)
        cv.imwrite(tube_strip_path, tube_strip)
        # cv.imwrite(annotated_strip_path, annotated_strip)

        # Draw label and rectangle
        cv.putText(
            img, f"tube{i}", (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1
        )
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # To annotate the number of bottles in the image
    cv.putText(
        img,
        f"no. of tubes: {i+1}",
        (10, 740),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (1, 255, 0),
    )

    # Save the annotated image
    cv.imwrite(output_path, img)
