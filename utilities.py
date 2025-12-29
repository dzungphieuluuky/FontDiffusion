def print_font_glyph_counts(fonts_dir: str):
    """
    Print the total number of glyphs (supported Unicode characters) for each font in fonts_dir.
    """
    from fontTools.ttLib import TTFont
    import glob
    import os

    font_extensions = ('.ttf', '.otf', '.TTF', '.OTF')
    font_files = [
        os.path.join(fonts_dir, f)
        for f in os.listdir(fonts_dir)
        if f.endswith(font_extensions)
    ]

    print(f"\nFont glyph count summary for directory: {fonts_dir}")
    for font_path in font_files:
        try:
            font = TTFont(font_path)
            cmap = font.getBestCmap()
            num_glyphs = len(cmap)
            font_name = os.path.splitext(os.path.basename(font_path))[0]
            print(f"  {font_name}: {num_glyphs} glyphs")
        except Exception as e:
            print(f"  {font_path}: Failed to read ({e})")
if __name__ == "__main__":
    directory = "fonts"
    print_font_glyph_counts(directory)