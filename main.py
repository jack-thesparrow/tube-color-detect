import cv2 as cv
import shutil
import numpy as np
import os
import argparse
from modules.imageProcess import crop_img, load_and_preprocess
from modules.bottleProcess import bottle_annotation, find_bottle_contour
from modules.tube_analyzer import analyze_tubes


def process_image(
    image_path, output_folder, tubes_out_folder, crop_x, crop_y, crop_w, crop_h
):
    img, hsv, eroded = load_and_preprocess(
        image_path,
        crop_w=crop_w,
        crop_h=crop_h,
        x_start=crop_x,
        y_start=crop_y,
    )
    contours = find_bottle_contour(eroded)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}_annotated.png")
    bottle_annotation(contours, img, output_path, tubes_out_folder, base_name)
    print(f"✅ Processed and saved: {output_path}")
    return base_name  # Needed for saving JSON


def main(input_path):
    output_folder = "output"
    tubes_out_folder = "tubes"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(tubes_out_folder, exist_ok=True)

    # Cropping parameters
    crop_x = 330
    crop_y = 240
    crop_w = 540
    crop_h = 780

    processed_basenames = []

    if os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                full_path = os.path.join(input_path, filename)
                try:
                    base = process_image(
                        full_path,
                        output_folder,
                        tubes_out_folder,
                        crop_x,
                        crop_y,
                        crop_w,
                        crop_h,
                    )
                    processed_basenames.append(base)
                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")
    elif os.path.isfile(input_path):
        try:
            base = process_image(
                input_path,
                output_folder,
                tubes_out_folder,
                crop_x,
                crop_y,
                crop_w,
                crop_h,
            )
            processed_basenames.append(base)
        except Exception as e:
            print(f"❌ Error processing {input_path}: {e}")
    else:
        print("❌ Error: Invalid input path.")
        return

    # === Analyze tubes and write one JSON per image ===
    for base_name in processed_basenames:
        json_path = os.path.join(
            os.path.dirname(__file__), f"{base_name}_analysis.json"
        )
        analyze_tubes(
            tubes_out_folder,
            save_json=True,
            filter_prefix=base_name,
            combined_json_path=json_path,
        )

    # === Clean up folders ===
    try:
        shutil.rmtree(tubes_out_folder)
        print(f"🧹 Deleted folder: {tubes_out_folder}")
    except Exception as e:
        print(f"⚠️ Error deleting {tubes_out_folder}: {e}")

    try:
        shutil.rmtree(output_folder)
        print(f"🧹 Deleted folder: {output_folder}")
    except Exception as e:
        print(f"⚠️ Error deleting {output_folder}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image(s) and analyze tubes.")
    parser.add_argument("input", nargs="?", help="Path to image file or folder.")
    args = parser.parse_args()

    if not args.input:
        print("❌ Please specify an image file or folder.")
        print("Usage: python main.py <image_or_folder_path>")
        exit(1)

    main(args.input)
