import cv2 as cv
from sklearn.cluster import KMeans
import numpy as np
import os


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
        tube_strip = tube[3:-3, 15:-15]
        tube_strip = segment_image(tube_strip)
        tube_path = os.path.join(tubes_out_path, f"{base_name}_tube{i}.png")
        tube_strip_path = os.path.join(tubes_out_path, f"{base_name}_tube_strip{i}.png")
        # cv.imwrite(tube_path, tube)
        cv.imwrite(tube_strip_path, tube_strip)

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
