import json
import math
import os.path
import threading
from typing import List, Optional, Dict, Tuple, Iterable

import ahocorasick
from ovos_utils.parse import fuzzy_match, MatchStrategy

from ovos_color_parser.core import (srgb8_to_linear, linear_to_srgb8, blend_linear,
                                    GamutPolicy, fit_to_gamut, srgb8_distance)
from ovos_color_parser.models import Color, sRGBAColor, HLSColor, sRGBAColorPalette


def color_distance(color_a: Color, color_b: Color) -> float:
    if not isinstance(color_a, sRGBAColor):
        color_a = color_a.as_rgb
    if not isinstance(color_b, sRGBAColor):
        color_b = color_b.as_rgb
    return srgb8_distance((color_a.r, color_a.g, color_a.b),
                          (color_b.r, color_b.g, color_b.b))


def closest_color(color: Color, color_opts: List[Color]) -> Color:
    color_opts = [c if isinstance(c, sRGBAColor) else c.as_rgb for c in color_opts]
    scores = {c: color_distance(color, c) for c in color_opts}
    return min(scores, key=lambda k: scores[k])


_RES_ROOT = f"{os.path.dirname(__file__)}/res"


def _resolve_lang_dir(lang: str) -> Optional[str]:
    """Resolve a requested language tag to an existing ``res/<locale>`` directory.

    The shipped resources are stored under full BCP-47 locale folders
    (``en-US``, ``de-DE``, ...). Instead of blindly stripping the region,
    pick the best matching directory:

    1. exact case-insensitive full-locale match (``en-us`` -> ``en-US``),
    2. same primary language subtag (``en`` / ``en-GB`` -> ``en-US``).

    Returns the absolute directory path, or ``None`` if nothing matches.
    """
    if not lang:
        return None
    available = [d for d in os.listdir(_RES_ROOT)
                 if os.path.isdir(f"{_RES_ROOT}/{d}")]
    by_lower = {d.lower(): d for d in available}

    requested = lang.lower().replace("_", "-")
    # 1) exact full-locale match
    if requested in by_lower:
        return f"{_RES_ROOT}/{by_lower[requested]}"
    # 2) same primary subtag (prefer the bare-subtag dir if present,
    #    then any locale sharing that subtag)
    primary = requested.split("-")[0]
    if primary in by_lower:
        return f"{_RES_ROOT}/{by_lower[primary]}"
    for d in sorted(available):
        if d.lower().split("-")[0] == primary:
            return f"{_RES_ROOT}/{d}"
    return None


def _load_color_json(lang: str) -> Iterable[Dict[str, str]]:
    p = _resolve_lang_dir(lang)
    if not p:
        return
    for wordlist in os.listdir(p):
        if not wordlist.endswith(".json") or wordlist == "color_descriptors.json":
            continue
        with open(f"{p}/{wordlist}") as f:
            words = json.load(f)
            yield words


def lookup_name(color: Color, lang: str = "en") -> str:
    if not isinstance(color, sRGBAColor):
        color = color.as_rgb
    for colorlist in _load_color_json(lang):
        if color.hex_str in colorlist:
            return colorlist[color.hex_str]
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
        res_dir = _resolve_lang_dir(lang)
        if not res_dir:
            return {}
        path = f"{res_dir}/object_colors.json"
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
            for colorlist in _load_color_json(lang):
                for hex_str, name in colorlist.items():
                    automaton.add_word(_norm(name), hex_str)
            if len(automaton):
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
            if len(automaton):
                automaton.make_automaton()
            cls._object_automatons[lang] = automaton
        return automaton

    @staticmethod
    def match_automaton(automaton, description) -> List[str]:
        # an automaton built from an empty wordlist (e.g. a locale without
        # object_colors.json) is never converted via make_automaton()
        if len(automaton) == 0:
            return []
        return [hex_str for _, hex_str in automaton.iter(_norm(description))]

    @classmethod
    def match_color_automaton(cls, description: str, lang: str = "en",
                              strategy: MatchStrategy = MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY,
                              fuzzy: bool = False) -> List[Tuple[HLSColor, float]]:
        automaton = ColorMatcher.load_color_automaton(lang)
        candidates = []
        weights = []
        for color_dict in _load_color_json(lang):
            if fuzzy:
                for h, n in color_dict.items():
                    s = fuzzy_match(_norm(n), _norm(description), strategy=MatchStrategy.TOKEN_SET_RATIO)
                    if s >= 0.8:
                        s = fuzzy_match(_norm(n), _norm(description), strategy=strategy)
                        if s >= 0.15:
                            #print(f"DEBUG: matched fuzzy color -> {(n, h, s)}")
                            weights.append(s)
                            try:
                                candidates.append(HLSColor.from_hex_str(h, name=n))
                            except ValueError as e:
                                #print(f"DEBUG: {e}")
                                pass
            else:
                hex_strs = cls.match_automaton(automaton, description)
                for hex_str in hex_strs:
                    if hex_str not in color_dict:
                        continue
                    name = color_dict[hex_str]
                    s = fuzzy_match(name, description, strategy=strategy)
                    if s >= 0.15:
                        # print(f"DEBUG: matched color -> {(name, hex_str, s)}")
                        weights.append(s)
                        candidates.append(HLSColor.from_hex_str(hex_str, name=name))
        #print(candidates, weights)
        return list(zip(candidates, weights))

    @classmethod
    def match_object_automaton(cls, description: str, lang: str = "en",
                               strategy: MatchStrategy = MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY
                               ) -> List[Tuple[HLSColor, float]]:
        obj_dict = cls._get_object_colors(lang)
        automaton = ColorMatcher.load_object_automaton(lang)
        hex_strs = cls.match_automaton(automaton, description)
        candidates = []
        weights = []
        for hex_s in hex_strs:
            if hex_s not in obj_dict:
                continue
            name = obj_dict[hex_s]
            weights.append(fuzzy_match(name, description, strategy=strategy))
            candidates.append(HLSColor.from_hex_str(hex_s, name=name))
        return list(zip(candidates, weights))


def _get_color_adjectives(lang: str) -> Dict[str, List[str]]:
    res_dir = _resolve_lang_dir(lang)
    if not res_dir:
        return {}
    path = f"{res_dir}/color_descriptors.json"
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _adjust_color_attributes(color: Color, description: str, adjectives: dict) -> sRGBAColor:
    if not isinstance(color, HLSColor):
        color = color.as_hls

    # no descriptor wordlist for this locale -> nothing to adjust
    if not adjectives:
        return color.as_rgb

    description = description.lower().strip()

    def matches(key: str) -> bool:
        return any(word.lower() in description for word in adjectives.get(key, []))

    # Saturation adjustments with additive/subtractive control
    if matches("very_high_saturation"):
        color.s = min(1.0, color.s + 0.2)  # Increase saturation
    elif matches("high_saturation"):
        color.s = min(1.0, color.s + 0.1)
    elif matches("low_saturation"):
        color.s = max(0.0, color.s - 0.1)
    elif matches("very_low_saturation"):
        color.s = max(0.0, color.s - 0.2)

    # Brightness adjustments with gamma-like control
    if matches("very_high_brightness"):
        color.l = min(1.0, color.l + 0.2)
    elif matches("high_brightness"):
        color.l = min(1.0, color.l + 0.1)
    elif matches("low_brightness"):
        color.l = max(0.0, color.l - 0.1)
    elif matches("very_low_brightness"):
        color.l = max(0.0, color.l - 0.2)

    color = color.as_rgb

    # Opacity adjustments (alpha channel, 0-255)
    if matches("very_high_opacity"):
        color.a = min(255, round(color.a * 1.5))
    elif matches("high_opacity"):
        color.a = min(255, round(color.a * 1.2))
    elif matches("low_opacity"):
        color.a = max(0, round(color.a * 0.7))
    elif matches("very_low_opacity"):
        color.a = max(0, round(color.a * 0.5))

    # Temperature adjustments using RGB tinting (channels are 0-255)
    if matches("very_high_temperature"):
        color.r = min(255, color.r + 26)
        color.g = max(0, color.g - 13)  # Add warmth by reducing green/blue tones
    elif matches("high_temperature"):
        color.r = min(255, color.r + 13)
    elif matches("low_temperature"):
        color.b = min(255, color.b + 13)  # Add coolness by increasing blue tones
    elif matches("very_low_temperature"):
        color.b = min(255, color.b + 26)

    return color


def palette_from_description(description: str, lang: str = "en",
                               strategy: MatchStrategy = MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY) -> sRGBAColorPalette:
    colors = [c for c, _ in ColorMatcher.match_color_automaton(description, lang, strategy, fuzzy=True)]
    #print(f"DEBUG: matched color names -> {[(_.name, _.hex_str) for _ in colors]}")
    return sRGBAColorPalette(colors=[_.as_rgb for _ in colors])


def color_from_description(description: str, lang: str = "en",
                           strategy: MatchStrategy = MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY,
                           cast_to_palette: bool = False,
                           fuzzy: bool = True) -> Optional[sRGBAColor]:
    candidates: List[HLSColor] = []
    weights: List[float] = []

    # step 1 - match color db
    for color, conf in ColorMatcher.match_color_automaton(description, lang, strategy, fuzzy=fuzzy):
        candidates.append(color)
        weights.append(conf)

    # Step 2 - match object names
    for color, conf in ColorMatcher.match_object_automaton(description, lang, strategy):
        candidates.append(color)
        weights.append(conf)

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
        #print(f"DEBUG: candidate colors: {[(_.name, _.hex_str) for _ in candidates]}")
        c = closest_color(c, candidates)
        #print(f"DEBUG: closest candidate color: {c} {c.hex_str}")

    c.description = description
    return c


def average_colors(colors: List[Color], weights: Optional[List[float]] = None) -> HLSColor:
    """Weighted mix of colors, returned as an :class:`HLSColor`.

    The mix is computed in **linear-light sRGB** (see :mod:`core.space`), which is
    how light physically combines. Averaging in gamma-encoded HLS instead darkens
    and desaturates the result — the classic "muddy blend" artefact. Named inputs
    are recorded in the description without leaking Python container internals.
    """
    if not colors:
        raise ValueError("colors must be a non-empty list")
    if weights is not None and len(weights) != len(colors):
        raise ValueError("weights must have the same length as colors")

    rgbs = [c if isinstance(c, sRGBAColor) else c.as_rgb for c in colors]
    lin = blend_linear([srgb8_to_linear(c.r, c.g, c.b, c.a) for c in rgbs], weights)
    r, g, b, a = linear_to_srgb8(lin)

    names = [c.name for c in colors if getattr(c, "name", None)]
    desc = "Weighted average of " + (", ".join(names) if names else f"{len(colors)} colors")
    return sRGBAColor(r, g, b, a, description=desc).as_hls


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
    ccolor = sRGBAColor.from_hex_str("#000000", name="black") \
        if yiq > 125 else sRGBAColor.from_hex_str("#ffffff", name="white")
    return ccolor


def is_hex_code_valid(hex_code: str) -> bool:
    """Validate whether the input string is a valid 3 or 6 digit hex color code."""
    hex_code = hex_code.lstrip("#")
    if len(hex_code) not in (3, 6):
        return False
    try:
        int(hex_code, 16)
    except ValueError:
        return False
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
