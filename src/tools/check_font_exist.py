import cv2
import copy
import pygame
from pygame import freetype
import numpy as np
from PIL import Image
from fontTools.ttLib import TTFont
from pathlib import Path


def is_char_in_font(font_path, char):
    TTFont_font = TTFont(font_path)
    cmap = TTFont_font["cmap"]
    for subtable in cmap.tables:
        if ord(char) in subtable.cmap:
            return True
    return False


def load_ttf(ttf_path, fsize=128) -> freetype.Font:
    pygame.init()

    font = freetype.Font(ttf_path, size=fsize)
    return font


def ttf2im(font, char, fsize=128):
    try:
        surface, _ = font.render(char)
    except:
        print("No glyph for char {}".format(char))
        return
    bg = np.full((fsize, fsize), 255)
    imo = pygame.surfarray.pixels_alpha(surface).transpose(1, 0)
    imo = 255 - np.array(Image.fromarray(imo))
    im = copy.deepcopy(bg)
    h, w = imo.shape[:2]
    if h > fsize:
        h, w = fsize, round(w * fsize / h)
        imo = cv2.resize(imo, (w, h))
    if w > fsize:
        h, w = round(h * fsize / w), fsize
        imo = cv2.resize(imo, (w, h))
    x, y = round((fsize - w) / 2), round((fsize - h) / 2)
    im[y : h + y, x : x + w] = imo
    pil_im = Image.fromarray(im.astype("uint8")).convert("RGB")

    return pil_im


def print_font_glyph_counts(fonts_dir: Path | str) -> None:
    """
    Print the number of glyphs (supported Unicode characters) for each font in a directory.

    Parameters
    ----------
    fonts_dir : Path | str
        Directory containing font files (.ttf, .otf).
    """
    from fontTools.ttLib import TTFont

    font_extensions = {".ttf", ".otf", ".ttf", ".otf"}

    font_files = [p for p in fonts_dir.iterdir() if p.suffix.lower() in font_extensions]

    print(f"\nFont glyph count summary for directory: {fonts_dir}")
    for font_path in font_files:
        try:
            font = TTFont(font_path)
            cmap = font.getBestCmap()
            num_glyphs = len(cmap)
            font_name = font_path.stem
            print(f"  {font_name}: {num_glyphs} glyphs")
        except Exception as e:
            print(f"  {font_path.name}: Failed to read ({e})")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Check and print the number of glyphs in each font file within a directory."
    )
    parser.add_argument(
        "fonts_dir",
        type=Path,
        help="Directory containing font files (.ttf, .otf) to check.",
    )

    args = parser.parse_args()
    print_font_glyph_counts(args.fonts_dir)