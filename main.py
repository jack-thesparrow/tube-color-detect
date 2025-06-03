import cv2 as cv
import numpy as np
import os
from modules.imageProcess import crop_img, load_and_preprocess
from modules.bottleProcess import bottle_annotation, find_bottle_contour


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
    main("backup")
