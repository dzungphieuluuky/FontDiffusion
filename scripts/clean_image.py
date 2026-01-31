import cv2
import numpy as np
import argparse
import os
from pathlib import Path


def clean_and_center_character(image_path, output_path, target_size=256):
    """
    Cleans, binarizes, and centers a character image for FontDiffuser.
    """
    # 1. Load Image
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    # 2. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Denoising (Vital for grainy scans like your 'thanh' and 'dao' series)
    denoised = cv2.medianBlur(gray, 3)

    # 4. Adaptive Binarization (Handles yellowed/darkened paper)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    # 5. Morphology Cleanup (Removes stray speckles)
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 6. Bounding Box & Centering
    coords = cv2.findNonZero(binary)
    if coords is None:
        return False

    x, y, w, h = cv2.boundingRect(coords)
    character_crop = binary[y : y + h, x : x + w]

    # Calculate resizing to fit canvas while maintaining aspect ratio
    padding = 30
    max_glyph_dim = target_size - (padding * 2)

    ratio = min(max_glyph_dim / w, max_glyph_dim / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized_char = cv2.resize(
        character_crop, (new_w, new_h), interpolation=cv2.INTER_AREA
    )

    # Create black canvas and paste character in the middle
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    x_off, y_off = (target_size - new_w) // 2, (target_size - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized_char

    # Invert: Models usually prefer Black text on White background
    final_output = cv2.bitwise_not(canvas)

    cv2.imwrite(str(output_path), final_output)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Clean Han-Nom characters for FontDiffuser."
    )
    parser.add_argument(
        "input_folder", type=str, help="Folder containing raw image crops"
    )
    parser.add_argument(
        "--output", type=str, default="cleaned_handwritten", help="Output folder name"
    )
    parser.add_argument(
        "--size", type=int, default=256, help="Target square size (default 256)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_folder)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Folder '{input_dir}' not found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif")
    files = [f for f in input_dir.iterdir() if f.suffix.lower() in valid_exts]

    print(f"Found {len(files)} images. Starting cleanup...")

    count = 0
    for f in files:
        out_file = output_dir / f"{f.stem}_cleaned.png"
        if clean_and_center_character(f, out_file, args.size):
            count += 1

    print(f"Done! {count} images cleaned and saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
