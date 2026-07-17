import json
import math
import os.path
import re
import threading
from typing import List, Optional, Dict, Tuple, Iterable

from ovos_utils.parse import fuzzy_match, MatchStrategy

from ovos_color_parser.core import (srgb8_to_linear, linear_to_srgb8, blend_linear,
                                    GamutPolicy, fit_to_gamut, srgb8_distance)
from ovos_color_parser.match import SubstringMatcher
from ovos_color_parser.models import Color, sRGBAColor, HLSColor, sRGBAColorPalette
from ovos_color_parser.vocab import (iter_color_dicts, load_palettes, palette_names)
from ovos_color_parser.vocab.loader import _resolve_lang_dir


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


def _load_color_json(lang: str) -> Iterable[Dict[str, str]]:
    """Locale color dicts, cached. Delegates to the vocab loader (kept as a thin
    wrapper for backward compatibility)."""
    return iter_color_dicts(lang)


def lookup_name(color: Color, lang: str = "en",
                namespace: Optional[str] = None, nearest: bool = False) -> str:
    """Name ``color`` from the known vocabularies.

    - ``namespace`` restricts the lookup to a single palette (e.g. ``"webcolors"``,
      ``"RAL_classic"``) instead of searching all of them.
    - Resolution order is deterministic (common palettes before niche catalogs),
      so the same hex always yields the same name.
    - With ``nearest=True`` an exact-miss falls back to the perceptually closest
      named color in scope instead of raising.

    Raises ``ValueError`` if nothing matches and ``nearest`` is False.
    """
    if not isinstance(color, sRGBAColor):
        color = color.as_rgb
    palettes = load_palettes(lang)
    if namespace is not None:
        if namespace not in palettes:
            raise ValueError(f"unknown color namespace: {namespace!r}. "
                             f"available: {palette_names(lang)}")
        palettes = {namespace: palettes[namespace]}

    for colorlist in palettes.values():
        if color.hex_str in colorlist:
            return colorlist[color.hex_str]

    if nearest:
        best_name, best_dist = None, None
        for colorlist in palettes.values():
            for hex_str, name in colorlist.items():
                try:
                    d = color_distance(color, sRGBAColor.from_hex_str(hex_str))
                except ValueError:
                    continue
                if best_dist is None or d < best_dist:
                    best_name, best_dist = name, d
        if best_name is not None:
            return best_name
    raise ValueError("Unnamed color")


# Arabic short vowels and other tashkeel are optional diacritics: the same word
# is written with or without them ("أَحْمَر" and "أحمر" are the same word). They
# occupy dedicated Arabic-script code points that never appear in other scripts,
# so removing them makes matching diacritic-insensitive without affecting any
# other language. Tatweel (kashida) is a purely cosmetic letter-stretching
# character and is dropped for the same reason.
_ARABIC_DIACRITICS = re.compile(
    "[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ࣓-ࣿـ]"
)


def _strip_arabic_diacritics(k: str) -> str:
    return _ARABIC_DIACRITICS.sub("", k)


def _norm(k):
    """
    Normalize a string by converting it to lowercase, replacing hyphens and underscores with spaces,
    stripping punctuation and whitespace, and removing Arabic diacritics so a word
    matches whether or not it is written with tashkeel.
    """
    k = _strip_arabic_diacritics(k)
    return k.lower().replace("-", " ").replace("_", " ").strip(" ,.!\n:;")


_DEFAULT_STRATEGY = MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY
_MIN_SCORE = 0.15  # discard weak name matches below this similarity
_MIN_FUZZY_LEN = 3  # names shorter than this are matched exactly, never fuzzily


def _build_automaton(color_dicts: Iterable[Dict[str, str]]) -> SubstringMatcher:
    automaton = SubstringMatcher()
    for colorlist in color_dicts:
        for hex_str, name in colorlist.items():
            automaton.add_word(_norm(name), hex_str)
    if len(automaton):
        automaton.make_automaton()
    return automaton


def _hits(automaton: SubstringMatcher, description: str) -> List[str]:
    # an automaton built from an empty wordlist is never made, so guard on len
    if len(automaton) == 0:
        return []
    return [hex_str for _, hex_str in automaton.iter(_norm(description))]


def _specificity(name: str) -> float:
    """Weight an exact match by how specific its name is: a longer, multi-word
    name ("moss green") describes the intent more precisely than a short generic
    one ("green"), so it should dominate the blend. Scoring by name length also
    means a real match no longer washes out on long input, unlike comparing the
    name against the whole sentence."""
    return float(max(1, len(_norm(name))))


def _exact_color_matches(color_dicts: List[Dict[str, str]], automaton: SubstringMatcher,
                         description: str, strategy: MatchStrategy) -> List[Tuple[HLSColor, float]]:
    # word-boundary spotting already guarantees each hit is a real, whole-word
    # occurrence, so matches are weighted by specificity rather than re-scored
    hits = _hits(automaton, description)
    out = []
    for color_dict in color_dicts:
        for hex_str in hits:
            if hex_str not in color_dict:
                continue
            name = color_dict[hex_str]
            out.append((HLSColor.from_hex_str(hex_str, name=name), _specificity(name)))
    return out


def _fuzzy_color_matches(color_dicts: List[Dict[str, str]], description: str,
                         strategy: MatchStrategy) -> List[Tuple[HLSColor, float]]:
    """Similarity scan that also catches compound names an exact spotter misses
    (e.g. "dunkles rot" ~ "dunkelrot").

    A cheap token-set gate rejects most names before the more expensive
    edit-distance score is computed, so this stays bounded in practice.
    """
    out = []
    norm_desc = _norm(description)
    for color_dict in color_dicts:
        for hex_str, name in color_dict.items():
            norm_name = _norm(name)
            # very short names (e.g. the two-letter دم "blood") fuzzy-match almost
            # any word that contains them; edit distance is meaningless at that
            # length, and exact word-boundary spotting already covers them
            if len(norm_name) < _MIN_FUZZY_LEN:
                continue
            gate = fuzzy_match(norm_name, norm_desc, strategy=MatchStrategy.TOKEN_SET_RATIO)
            if gate < 0.8:
                continue
            s = fuzzy_match(norm_name, norm_desc, strategy=strategy)
            if s >= _MIN_SCORE:
                try:
                    out.append((HLSColor.from_hex_str(hex_str, name=name), s))
                except ValueError:
                    pass
    return out


def _merge_matches(*groups: List[Tuple[HLSColor, float]]) -> List[Tuple[HLSColor, float]]:
    """Union matches from several passes, keeping the highest score per
    (hex, name) so the exact and fuzzy passes don't double-count a color."""
    best: Dict[Tuple[str, Optional[str]], Tuple[HLSColor, float]] = {}
    for group in groups:
        for color, score in group:
            key = (color.hex_str, color.name)
            if key not in best or score > best[key][1]:
                best[key] = (color, score)
    return list(best.values())


def _object_matches(obj_dict: Dict[str, str], automaton: SubstringMatcher,
                    description: str, strategy: MatchStrategy) -> List[Tuple[HLSColor, float]]:
    out = []
    for hex_s in _hits(automaton, description):
        if hex_s not in obj_dict:
            continue
        name = obj_dict[hex_s]
        out.append((HLSColor.from_hex_str(hex_s, name=name), _specificity(name)))
    return out


class ColorMatcher:
    """Spots color and object names in free text.

    Can be used statically (``ColorMatcher.match_color_automaton(...)``, backed by
    a per-language global cache) or instantiated with a fixed language — or with
    custom vocabularies — for isolated, injectable matching::

        matcher = ColorMatcher("en")
        matcher.match_colors("moss green")
        ColorMatcher("xx", color_palettes=[{"#FF0000": "rood"}]).match_colors("rood")

    Matching is exact-first: the substring automaton is the fast path, and the
    costlier fuzzy scan runs only when exact spotting finds nothing.
    """

    _color_automatons: Dict[str, SubstringMatcher] = {}
    _object_automatons: Dict[str, SubstringMatcher] = {}
    __lock = threading.Lock()

    def __init__(self, lang: str = "en",
                 color_palettes: Optional[Iterable[Dict[str, str]]] = None,
                 object_colors: Optional[Dict[str, str]] = None) -> None:
        self.lang = lang
        self._color_dicts = (list(color_palettes) if color_palettes is not None
                             else list(iter_color_dicts(lang)))
        self._object_dict = (dict(object_colors) if object_colors is not None
                             else self._get_object_colors(lang))
        self._color_automaton = _build_automaton(self._color_dicts)
        self._object_automaton = _build_automaton([self._object_dict])

    def match_colors(self, description: str,
                     strategy: MatchStrategy = _DEFAULT_STRATEGY,
                     fuzzy: bool = False) -> List[Tuple[HLSColor, float]]:
        exact = _exact_color_matches(self._color_dicts, self._color_automaton,
                                     description, strategy)
        if not fuzzy:
            return exact
        return _merge_matches(exact, _fuzzy_color_matches(self._color_dicts, description, strategy))

    def match_objects(self, description: str,
                      strategy: MatchStrategy = _DEFAULT_STRATEGY) -> List[Tuple[HLSColor, float]]:
        return _object_matches(self._object_dict, self._object_automaton, description, strategy)

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
    def load_color_automaton(cls, lang: str) -> SubstringMatcher:
        with cls.__lock:
            if lang not in cls._color_automatons:
                cls._color_automatons[lang] = _build_automaton(iter_color_dicts(lang))
        return cls._color_automatons[lang]

    @classmethod
    def load_object_automaton(cls, lang: str) -> SubstringMatcher:
        with cls.__lock:
            if lang not in cls._object_automatons:
                cls._object_automatons[lang] = _build_automaton([cls._get_object_colors(lang)])
        return cls._object_automatons[lang]

    @staticmethod
    def match_automaton(automaton, description) -> List[str]:
        return _hits(automaton, description)

    @classmethod
    def match_color_automaton(cls, description: str, lang: str = "en",
                              strategy: MatchStrategy = _DEFAULT_STRATEGY,
                              fuzzy: bool = False) -> List[Tuple[HLSColor, float]]:
        color_dicts = list(iter_color_dicts(lang))
        exact = _exact_color_matches(color_dicts, cls.load_color_automaton(lang),
                                     description, strategy)
        if not fuzzy:
            return exact
        return _merge_matches(exact, _fuzzy_color_matches(color_dicts, description, strategy))

    @classmethod
    def match_object_automaton(cls, description: str, lang: str = "en",
                               strategy: MatchStrategy = _DEFAULT_STRATEGY
                               ) -> List[Tuple[HLSColor, float]]:
        return _object_matches(cls._get_object_colors(lang),
                               cls.load_object_automaton(lang), description, strategy)


def _get_color_adjectives(lang: str) -> Dict[str, List[str]]:
    res_dir = _resolve_lang_dir(lang)
    if not res_dir:
        return {}
    path = f"{res_dir}/color_descriptors.json"
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _fit(color: sRGBAColor, gamut: GamutPolicy) -> sRGBAColor:
    """Bring ``color`` into the sRGB gamut using ``gamut`` (hue-preserving MAP,
    per-channel CLAMP, or REJECT). In-gamut colors are returned unchanged."""
    lin, _ = fit_to_gamut(srgb8_to_linear(color.r, color.g, color.b, color.a), gamut)
    r, g, b, a = linear_to_srgb8(lin)
    return sRGBAColor(r, g, b, a, name=color.name, description=color.description)


def _adjust_color_attributes(color: Color, description: str, adjectives: dict,
                             gamut: GamutPolicy = GamutPolicy.CLAMP) -> sRGBAColor:
    if not isinstance(color, HLSColor):
        color = color.as_hls

    # no descriptor wordlist for this locale -> nothing to adjust
    if not adjectives:
        return color.as_rgb

    # strip diacritics so a vowelled modifier ("غَامِق") matches its bare
    # descriptor entry, exactly as color names are matched
    description = _strip_arabic_diacritics(description).lower().strip()

    def matches(key: str) -> bool:
        # match descriptor words on word boundaries so "light" fires on "light
        # blue" but not on "delight", and multi-word cues still match verbatim
        for word in adjectives.get(key, []):
            w = _strip_arabic_diacritics(word).lower().strip()
            if w and re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", description):
                return True
        return False

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

    return _fit(color, gamut)


def palette_from_description(description: str, lang: str = "en",
                               strategy: MatchStrategy = MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY) -> sRGBAColorPalette:
    colors = [c for c, _ in ColorMatcher.match_color_automaton(description, lang, strategy, fuzzy=True)]
    #print(f"DEBUG: matched color names -> {[(_.name, _.hex_str) for _ in colors]}")
    return sRGBAColorPalette(colors=[_.as_rgb for _ in colors])


def color_from_description(description: str, lang: str = "en",
                           strategy: MatchStrategy = MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY,
                           cast_to_palette: bool = False,
                           fuzzy: bool = True,
                           gamut: GamutPolicy = GamutPolicy.CLAMP) -> Optional[sRGBAColor]:
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
                                 _get_color_adjectives(lang), gamut)
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
