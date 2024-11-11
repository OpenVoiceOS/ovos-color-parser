import json
import math
import os.path
import threading
from typing import List, Optional, Dict, Tuple

import ahocorasick
from colorspacious import deltaE
from ovos_utils.parse import fuzzy_match, MatchStrategy

from ovos_color_parser.models import Color, sRGBAColor, HLSColor


def color_distance(color_a: Color, color_b: Color) -> float:
    if not isinstance(color_a, sRGBAColor):
        color_a = color_a.as_rgb
    if not isinstance(color_b, sRGBAColor):
        color_b = color_b.as_rgb
    return float(deltaE([color_a.r, color_a.g, color_a.b],
                        [color_b.r, color_b.g, color_b.b],
                        input_space="sRGB255"))


def closest_color(color: Color, color_opts: List[Color]) -> Color:
    color_opts = [c if isinstance(c, sRGBAColor) else c.as_rgb for c in color_opts]
    scores = {c: color_distance(color, c) for c in color_opts}
    return min(scores, key=lambda k: scores[k])


_COLOR_DATA: Dict[str, Dict[str, str]] = {}
__lock = threading.Lock()


def _load_color_json(lang: str) -> Dict[str, str]:
    global _COLOR_DATA
    with __lock:
        lang = lang.lower().split("-")[0]
        data = _COLOR_DATA.get(lang, {})
        if not data:
            path = f"{os.path.dirname(__file__)}/res/{lang}/colors.json"
            if os.path.isfile(path):
                with open(path) as f:
                    _COLOR_DATA[lang] = data = json.load(f)
    return data


def lookup_name(color: Color, lang: str = "en") -> str:
    if not isinstance(color, sRGBAColor):
        color = color.as_rgb
    data = _load_color_json(lang)
    if color.hex_str in data:
        return data[color.hex_str]
    raise ValueError("Unnamed color")


def _norm(k):
    """
    Normalize a string by converting it to lowercase, replacing hyphens and underscores with spaces,
    and stripping punctuation and whitespace characters.
    """
    return k.lower().replace("-", " ").replace("_", " ").strip(" ,.!\n:;")


class ColorMatcher:
    _color_automatons: Dict[str, ahocorasick.Automaton] = {}
    _object_automatons: Dict[str, ahocorasick.Automaton] = {}
    __lock = threading.Lock()

    @staticmethod
    def _get_object_colors(lang: str) -> Dict[str, str]:
        lang = lang.lower().split("-")[0]
        path = f"{os.path.dirname(__file__)}/res/{lang}/object_colors.json"
        if not os.path.isfile(path):
            return {}
        with open(path) as f:
            return json.load(f)

    @classmethod
    def load_color_automaton(cls, lang: str) -> ahocorasick.Automaton:
        with cls.__lock:
            if lang in cls._color_automatons:
                return cls._color_automatons[lang]
            automaton = ahocorasick.Automaton()
            for hex_str, name in _load_color_json(lang).items():
                automaton.add_word(_norm(name), hex_str)
            automaton.make_automaton()
            cls._color_automatons[lang] = automaton
        return automaton

    @classmethod
    def load_object_automaton(cls, lang: str) -> ahocorasick.Automaton:
        with cls.__lock:
            if lang in cls._object_automatons:
                return cls._object_automatons[lang]
            automaton = ahocorasick.Automaton()
            for hex_str, name in cls._get_object_colors(lang).items():
                automaton.add_word(_norm(name), hex_str)
            automaton.make_automaton()
            cls._object_automatons[lang] = automaton
        return automaton

    @staticmethod
    def match_automaton(automaton, description, data_dict, strategy):
        candidates = []
        weights = []
        for _, hex_str in automaton.iter(description):
            name = data_dict[hex_str]
            weights.append(fuzzy_match(name, description, strategy=strategy))
            candidates.append(HLSColor.from_hex_str(hex_str, name=name))
        return candidates, weights

    @classmethod
    def match_color_automaton(cls, description: str, lang: str = "en",
                              strategy: MatchStrategy = MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY):
        color_dict = _load_color_json(lang)
        automaton = ColorMatcher.load_color_automaton(lang)
        return cls.match_automaton(automaton, description, color_dict, strategy)

    @classmethod
    def match_object_automaton(cls, description: str, lang: str = "en",
                               strategy: MatchStrategy = MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY):
        obj_dict = cls._get_object_colors(lang)
        automaton = ColorMatcher.load_object_automaton(lang)
        return cls.match_automaton(automaton, description, obj_dict, strategy)


def _get_color_adjectives(lang: str) -> Dict[str, List[str]]:
    lang = lang.lower().split("-")[0]
    path = f"{os.path.dirname(__file__)}/res/{lang}/color_descriptors.json"
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _adjust_color_attributes(color: Color, description: str, adjectives: dict) -> sRGBAColor:
    if not isinstance(color, HLSColor):
        color = color.as_hls

    description = description.lower().strip()

    # Saturation adjustments with additive/subtractive control
    if any(word.lower() in description for word in adjectives["very_high_saturation"]):
        color.s = min(1.0, color.s + 0.2)  # Increase saturation
    elif any(word.lower() in description for word in adjectives["high_saturation"]):
        color.s = min(1.0, color.s + 0.1)
    elif any(word.lower() in description for word in adjectives["low_saturation"]):
        color.s = max(0.0, color.s - 0.1)
    elif any(word.lower() in description for word in adjectives["very_low_saturation"]):
        color.s = max(0.0, color.s - 0.2)

    # Brightness adjustments with gamma-like control
    if any(word.lower() in description for word in adjectives["very_high_brightness"]):
        color.l = min(1.0, color.l + 0.2)
    elif any(word.lower() in description for word in adjectives["high_brightness"]):
        color.l = min(1.0, color.l + 0.1)
    elif any(word.lower() in description for word in adjectives["low_brightness"]):
        color.l = max(0.0, color.l - 0.1)
    elif any(word.lower() in description for word in adjectives["very_low_brightness"]):
        color.l = max(0.0, color.l - 0.2)

    # Opacity adjustments
    if any(word.lower() in description for word in adjectives["very_high_opacity"]):
        color.a = min(1.0, color.a * 1.5)
    elif any(word.lower() in description for word in adjectives["high_opacity"]):
        color.a = min(1.0, color.a * 1.2)
    elif any(word.lower() in description for word in adjectives["low_opacity"]):
        color.a = max(0.0, color.a * 0.7)
    elif any(word.lower() in description for word in adjectives["very_low_opacity"]):
        color.a = max(0.0, color.a * 0.5)

    # Temperature adjustments using RGB tinting
    color = color.as_rgb
    if any(word.lower() in description for word in adjectives["very_high_temperature"]):
        color.r = min(1.0, color.r + 0.1)
        color.g = max(0.0, color.g - 0.05)  # Add warmth by reducing blue tones
    elif any(word.lower() in description for word in adjectives["high_temperature"]):
        color.r = min(1.0, color.r + 0.05)
    elif any(word.lower() in description for word in adjectives["low_temperature"]):
        color.b = min(1.0, color.b + 0.05)  # Add coolness by increasing blue tones
    elif any(word.lower() in description for word in adjectives["very_low_temperature"]):
        color.b = min(1.0, color.b + 0.1)

    return color


def color_from_description(description: str, lang: str = "en",
                           strategy: MatchStrategy = MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY,
                           cast_to_palette: bool = False) -> Optional[sRGBAColor]:
    candidates: List[HLSColor] = []
    weights: List[float] = []

    # step 1 - match color db
    c, w = ColorMatcher.match_color_automaton(description, lang, strategy)
    # print(f"DEBUG: matched color names -> {c}")
    candidates += c
    weights += w

    # Step 2 - match object names
    c, w = ColorMatcher.match_object_automaton(description, lang, strategy)
    # print(f"DEBUG: matched object name -> {c}")
    candidates += c
    weights += w

    # Step 3 - select base color
    if candidates:
        c = average_colors(candidates, weights)
        # c2 = closest_color(c, candidates)
        # print(f"DEBUG: closest candidate color: {c2}:{c2.hex_str}")
    else:
        return None

    # Step 4 - match luminance/saturation keywords
    c = _adjust_color_attributes(c, description,
                                 _get_color_adjectives(lang))
    c.name = description.title()

    # do not invent colors
    if cast_to_palette:
        c = closest_color(c, candidates)

    c.description = description
    return c


def average_colors(colors: List[Color], weights: Optional[List[float]] = None) -> HLSColor:
    colors = [c if isinstance(c, HLSColor) else c.as_hls for c in colors]
    weights = weights or [1 / len(colors) for c in colors]

    # Step 1: Weighted averages for Lightness and Saturation
    total_weight = sum(weights)
    avg_l = sum(c.l * w for c, w in zip(colors, weights)) / total_weight
    avg_s = sum(c.s * w for c, w in zip(colors, weights)) / total_weight

    # Step 2: Weighted circular mean for Hue
    sin_sum = sum(math.sin(math.radians(c.h)) * w for c, w in zip(colors, weights))
    cos_sum = sum(math.cos(math.radians(c.h)) * w for c, w in zip(colors, weights))
    avg_h = int(math.degrees(math.atan2(sin_sum, cos_sum)) % 360)  # Ensure hue is in [0, 360)

    # Return new averaged HLSColor
    return HLSColor(h=avg_h, l=avg_l, s=avg_s,
                    description=f"Weighted average: {set(zip([c.name for c in colors], weights))}")


def convert_K_to_RGB(colour_temperature: int) -> sRGBAColor:
    """
    Taken from: http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code/
    Converts from K to RGB, algorithm courtesy of
    http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code/
    """
    # range check
    if colour_temperature < 1000 or colour_temperature > 40000:
        raise ValueError("color temperature out of range, only values between 1000 and 40000 supported")

    tmp_internal = colour_temperature / 100.0

    # red
    if tmp_internal <= 66:
        red = 255
    else:
        tmp_red = 329.698727446 * math.pow(tmp_internal - 60, -0.1332047592)
        if tmp_red < 0:
            red = 0
        elif tmp_red > 255:
            red = 255
        else:
            red = tmp_red

    # green
    if tmp_internal <= 66:
        tmp_green = 99.4708025861 * math.log(tmp_internal) - 161.1195681661
        if tmp_green < 0:
            green = 0
        elif tmp_green > 255:
            green = 255
        else:
            green = tmp_green
    else:
        tmp_green = 288.1221695283 * math.pow(tmp_internal - 60, -0.0755148492)
        if tmp_green < 0:
            green = 0
        elif tmp_green > 255:
            green = 255
        else:
            green = tmp_green

    # blue
    if tmp_internal >= 66:
        blue = 255
    elif tmp_internal <= 19:
        blue = 0
    else:
        tmp_blue = 138.5177312231 * math.log(tmp_internal - 10) - 305.0447927307
        if tmp_blue < 0:
            blue = 0
        elif tmp_blue > 255:
            blue = 255
        else:
            blue = tmp_blue

    return sRGBAColor(int(red), int(green), int(blue), description=f"{colour_temperature}K")


def get_contrasting_black_or_white(hex_code: str) -> sRGBAColor:
    """Get a contrasting black or white color for text display.

    This gets calculated based off the input color using the YIQ system.
    https://en.wikipedia.org/wiki/YIQ

    Args:
        hex_code of base color

    Returns:
        black or white as a hex_code
    """
    color = sRGBAColor.from_hex_str(hex_code)
    yiq = ((color.r * 299) + (color.g * 587) + (color.b * 114)) / 1000
    ccolor = sRGBAColor.from_hex_str("#000000", name="white") \
        if yiq > 125 else sRGBAColor.from_hex_str("#ffffff", name="black")
    return ccolor


def is_hex_code_valid(hex_code: str) -> bool:
    """Validate whether the input string is a valid hex color code."""
    # TODO expand to validate 3 char codes.
    hex_code = hex_code.lstrip("#")
    try:
        assert len(hex_code) == 6
        int(hex_code, 16)
    except (AssertionError, ValueError):
        return False
    else:
        return True


def rgb_to_cmyk(r, g, b, cmyk_scale=100, rgb_scale=255) -> Tuple[float, float, float, float]:
    if (r, g, b) == (0, 0, 0):
        # black
        return 0, 0, 0, cmyk_scale

    # rgb [0,255] -> cmy [0,1]
    c = 1 - r / rgb_scale
    m = 1 - g / rgb_scale
    y = 1 - b / rgb_scale

    # extract out k [0, 1]
    min_cmy = min(c, m, y)
    c = (c - min_cmy) / (1 - min_cmy)
    m = (m - min_cmy) / (1 - min_cmy)
    y = (y - min_cmy) / (1 - min_cmy)
    k = min_cmy

    # rescale to the range [0,CMYK_SCALE]
    return c * cmyk_scale, m * cmyk_scale, y * cmyk_scale, k * cmyk_scale


def cmyk_to_rgb(c, m, y, k, cmyk_scale=100, rgb_scale=255) -> Tuple[int, int, int]:
    r = rgb_scale * (1.0 - c / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale))
    g = rgb_scale * (1.0 - m / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale))
    b = rgb_scale * (1.0 - y / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale))
    return int(r), int(g), int(b)
