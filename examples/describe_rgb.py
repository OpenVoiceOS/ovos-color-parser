"""Reverse direction: turn a raw color into words you can speak or print.

Given an RGB tuple or hex string (from a color picker, a sensor, a smart bulb,
an image), produce a natural-language color name. Handy for TTS ("the light is
now warm white"), accessibility labels, or alt-text.

    pip install ovos-color-parser
    python describe_rgb.py

With `nearest=True`, `lookup_name` snaps an arbitrary color to the perceptually
closest entry in a language's wordlist and returns its name, so any color — not
just exact wordlist hits — speaks correctly in any supported language.
"""
from ovos_color_parser import sRGBAColor, lookup_name, get_contrasting_black_or_white


def describe(rgb, lang: str = "en") -> str:
    color = sRGBAColor(*rgb)
    return lookup_name(color, lang=lang, nearest=True)


if __name__ == "__main__":
    samples = [
        (255, 0, 0),
        (30, 144, 255),
        (255, 219, 88),
        (46, 139, 87),
        (128, 0, 128),
        (245, 245, 220),
    ]

    print("RGB -> spoken color name (English)\n")
    for rgb in samples:
        hex_str = sRGBAColor(*rgb).hex_str
        print(f"  {str(rgb):18} {hex_str}  ->  {describe(rgb)!r}")

    print("\nSame colors, other languages\n")
    for lang in ("en", "de", "fr", "pt", "ru"):
        print(f"  {lang}: {describe((30, 144, 255), lang=lang)!r}")

    print("\nHex from a color picker -> a sentence for TTS\n")
    for hex_code in ("#2E8B57", "#FFB6C1", "#36454F"):
        color = sRGBAColor.from_hex_str(hex_code)
        name = lookup_name(color, lang="en", nearest=True)
        ink = get_contrasting_black_or_white(hex_code).name  # readable label color
        print(f"  {hex_code}: \"The color is {name.lower()}.\"  (label text: {ink})")
