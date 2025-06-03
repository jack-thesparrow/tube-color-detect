import cv2 as cv
import os


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
        tube_path = os.path.join(tubes_out_path, f"{base_name}_tube{i}.png")
        cv.imwrite(tube_path, tube)

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
