"""UI / design tooling: map color descriptions to hex, RGB and LED values.

Turn typed or spoken color descriptions into concrete values you can feed to a
CSS variable, a theme file, an RGB LED strip or an image library -- no OVOS
required.

    pip install ovos-color-parser
    python ui_theming.py
"""
from ovos_color_parser import (color_from_description, sRGBAColor,
                               get_contrasting_black_or_white, convert_K_to_RGB)


def to_css_theme(spec: dict, lang: str = "en") -> dict:
    """{"--accent": "vivid teal"} -> {"--accent": "#0aa4a4"}."""
    theme = {}
    for var, description in spec.items():
        color = color_from_description(description, lang=lang)
        if color is not None:
            theme[var] = color.hex_str.lower()
    return theme


def to_led_rgb(description: str, lang: str = "en"):
    """A description -> an (r, g, b) tuple ready for a NeoPixel / WLED call."""
    color = color_from_description(description, lang=lang)
    return None if color is None else (color.r, color.g, color.b)


if __name__ == "__main__":
    spec = {
        "--bg": "very dark blue",
        "--surface": "warm gray",
        "--accent": "vivid teal",
        "--danger": "bright red",
    }
    print("CSS theme from descriptions:\n")
    theme = to_css_theme(spec)
    for var, hex_str in theme.items():
        text = get_contrasting_black_or_white(hex_str).name
        print(f"  {var:12}: {hex_str}   (readable text: {text})")

    print("\nLED strip values:\n")
    for phrase in ["moss green", "hot pink", "amber", "electric blue"]:
        print(f"  {phrase!r:16} -> setPixelColor{to_led_rgb(phrase)}")

    print("\nWhite-balance a bulb by color temperature (Kelvin):\n")
    for k in (2700, 4000, 6500):
        c = convert_K_to_RGB(k)
        print(f"  {k}K -> {c.hex_str} rgb({c.r}, {c.g}, {c.b})")

    print("\nSame theming flow in Spanish:\n")
    for var, hex_str in to_css_theme({"--acento": "verde vivo",
                                      "--fondo": "azul oscuro"}, lang="es").items():
        print(f"  {var:12}: {hex_str}")
