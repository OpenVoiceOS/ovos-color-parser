import json
import math
import os.path
from colorsys import rgb_to_hsv, hsv_to_rgb, hls_to_rgb, rgb_to_hls
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Tuple

import ahocorasick
from colorspacious import deltaE
from ovos_utils.parse import fuzzy_match, MatchStrategy


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


# Supported color spaces:
#  - RGB
#  - HSV
#  - HLS  <- all color operations are performed in this space
#  - Spectral (wave length)


@dataclass
class sRGBAColor:
    # Color defined in sRGB color space
    r: int
    g: int
    b: int
    a: int = 255
    name: Optional[str] = None
    description: Optional[str] = None

    def __hash__(self):
        return int(f"{self.r}{self.g}{self.b}")

    @property
    def as_spectral_color(self) -> 'SpectralColor':
        return self.as_hsv.as_spectral_color

    @property
    def as_hls(self) -> 'HLSColor':
        r = self.r / 255
        g = self.g / 255
        b = self.b / 255
        h, l, s = rgb_to_hls(r, g, b)
        return HLSColor(int(h * 360), l, min(1, s),
                        name=self.name,
                        description=self.description)

    @property
    def as_hsv(self) -> 'HSVColor':
        r = self.r / 255
        g = self.g / 255
        b = self.b / 255
        h, s, v = rgb_to_hsv(r, g, b)
        return HSVColor(int(h * 360), s, v,
                        name=self.name,
                        description=self.description)

    @property
    def hex_str(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}".upper()

    @staticmethod
    def from_hex_str(hex_str: str, name: Optional[str] = None, description: Optional[str] = None) -> 'sRGBAColor':
        if hex_str.startswith('#'):
            hex_str = hex_str[1:]
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        return sRGBAColor(r, g, b, name=name, description=description)

    def __post_init__(self):
        # Enforce hue values between 0 and 360
        if not (0 <= self.r <= 255) or not (0 <= self.r <= 255):
            raise ValueError("RGB values must be in the range 0 to 255")
        if not (0 <= self.g <= 255) or not (0 <= self.g <= 255):
            raise ValueError("RGB values must be in the range 0 to 255")
        if not (0 <= self.b <= 255) or not (0 <= self.b <= 255):
            raise ValueError("RGB values must be in the range 0 to 255")


@dataclass
class HSVColor:
    h: int
    s: float = 0.5
    v: float = 0.5
    name: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        # Enforce hue values between 0 and 360
        if not 0 <= self.h <= 360:
            raise ValueError("Hue values must be in the range 0 to 360")
        if not (0 <= self.s <= 1) or not (0 <= self.v <= 1):
            raise ValueError("Saturation and Value must be in the range 0 to 1")

    @property
    def as_spectral_color(self) -> 'SpectralColor':
        return HueRange(self.h, self.h, self.name, self.hex_str).as_spectral_color

    @property
    def as_rgb(self) -> 'sRGBAColor':
        r, g, b = hsv_to_rgb(self.h / 360, self.s, self.v)
        return sRGBAColor(int(r * 255), int(g * 255), int(b * 255),
                          name=self.name,
                          description=self.description)

    @property
    def as_hls(self) -> 'HLSColor':
        return self.as_rgb.as_hls

    @property
    def hex_str(self) -> str:
        return self.as_rgb.hex_str

    @staticmethod
    def from_hex_str(hex_str: str, name: Optional[str] = None, description: Optional[str] = None) -> 'HSVColor':
        return sRGBAColor.from_hex_str(hex_str, name, description).as_hsv


@dataclass
class HLSColor:
    h: int
    l: float = 0.5
    s: float = 0.5
    name: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        # Enforce hue values between 0 and 360
        if not 0 <= self.h <= 360:
            raise ValueError("Hue values must be in the range 0 to 360")
        if not (0 <= self.s <= 1) or not (0 <= self.l <= 1):
            raise ValueError("Saturation and Luminance must be in the range 0 to 1")

    @property
    def as_spectral_color(self) -> 'SpectralColor':
        return HueRange(self.h, self.h, self.name, self.hex_str).as_spectral_color

    @property
    def as_rgb(self) -> 'sRGBAColor':
        r, g, b = hls_to_rgb(self.h / 360, self.l, self.s)
        return sRGBAColor(int(r * 255), int(g * 255), int(b * 255),
                          name=self.name,
                          description=self.description)

    @property
    def as_hsv(self) -> 'HSVColor':
        return self.as_rgb.as_hsv

    @property
    def hex_str(self) -> str:
        return self.as_rgb.hex_str

    @staticmethod
    def from_hex_str(hex_str: str, name: Optional[str] = None, description: Optional[str] = None) -> 'HLSColor':
        return sRGBAColor.from_hex_str(hex_str, name, description).as_hls


# Just for fun, so we can map wavelens to colors
# physicists are huge nerds, so they might say "change the lamp wave lenght to X nanometers"
@dataclass
class SpectralColor:
    # Color defined via a wavelength range (in nanometers)
    wavelen_nm_min: int
    wavelen_nm_max: int
    hex_approximation: Optional[str] = None
    hue_approximation: Optional['HueRange'] = None
    name: Optional[str] = None

    @property
    def wavelen(self) -> int:
        return int((self.wavelen_nm_max + self.wavelen_nm_min) / 2)

    @staticmethod
    def _wavelength_to_hue(wavelen: int, palette: 'SpectralColorPalette') -> int:

        for color_term in palette.colors:
            hue_range = color_term.hue_approximation

            # Check if wavelen falls within the color's range
            if color_term.wavelen_nm_min <= wavelen <= color_term.wavelen_nm_max:
                # Interpolate the hue within this range based on the wavelen
                _span = color_term.wavelen_nm_max - color_term.wavelen_nm_min
                if _span == 0:
                    # Handle case for ranges with a single hue (no interpolation needed)
                    return hue_range.hue
                # Calculate the interpolated hue
                hue = hue_range.min_hue_approximation + ((wavelen - color_term.wavelen_nm_min) / _span) * (
                        hue_range.max_hue_approximation - hue_range.min_hue_approximation)
                return int(hue)

        # Raise an error if wavelength is out of any defined range in the palette
        raise ValueError("Wavelength is out of the defined spectral color palette.")

    @property
    def as_rgb(self) -> 'sRGBAColor':
        if self.hex_approximation:
            return sRGBAColor.from_hex_str(self.hex_approximation)
        if self.hue_approximation:
            return HSVColor(self.hue_approximation.hue).as_rgb
        return HSVColor(self._wavelength_to_hue(self.wavelen, ISCCNBSSpectralColorTerms)).as_rgb

    @property
    def as_hls(self) -> 'HLSColor':
        return self.as_rgb.as_hls

    @property
    def as_hsv(self) -> 'HSVColor':
        return self.as_rgb.as_hsv

    @staticmethod
    def from_rgb(r: int, g: int, b: int, name: Optional[str] = None,
                 description: Optional[str] = None) -> 'SpectralColor':
        return sRGBAColor(r, g, b, name=name, description=description).as_hsv.as_spectral_color

    @staticmethod
    def from_hsv(h: int, s: float, v: float, name: Optional[str] = None,
                 description: Optional[str] = None) -> 'SpectralColor':
        return HSVColor(h, s, v, name, description).as_spectral_color

    @staticmethod
    def from_hls(h: int, l: float, s: float, name: Optional[str] = None,
                 description: Optional[str] = None) -> 'SpectralColor':
        return HLSColor(h, l, s, name, description).as_spectral_color

    @staticmethod
    def from_hex_str(hex_str: str, name: Optional[str] = None,
                     description: Optional[str] = None) -> 'SpectralColor':
        return sRGBAColor.from_hex_str(hex_str, name, description).as_spectral_color


@dataclass
class HueRange:
    min_hue_approximation: int
    max_hue_approximation: int
    name: Optional[str] = None
    hex_approximation: Optional[str] = None

    @property
    def hue(self) -> int:
        return int((self.min_hue_approximation + self.max_hue_approximation) / 2)

    @property
    def as_spectral_color(self) -> 'SpectralColor':
        palette = ISCCNBSSpectralColorTerms
        # Compute min and max wavelengths based on hue range
        wavelen_min = self._hue_to_wavelength(self.min_hue_approximation, palette)
        wavelen_max = self._hue_to_wavelength(self.max_hue_approximation, palette)
        specolor = SpectralColor(wavelen_nm_min=wavelen_min, wavelen_nm_max=wavelen_max,
                                 hue_approximation=self, name=self.name,
                                 hex_approximation=self.hex_approximation)
        if not self.name:
            # avg wavlen
            nm = int((wavelen_max + wavelen_min) / 2)
            # the named color terms aren't continuous (not all wavlen have names)
            for color in palette.colors:
                if color.wavelen_nm_min <= nm <= color.wavelen_nm_max:
                    specolor.name = color.name
                    break
        return specolor

    @property
    def as_rgb(self) -> 'sRGBAColorPalette':
        return sRGBAColorPalette(colors=[])  # TODO

    @property
    def as_hls(self) -> 'HLSColorPalette':
        return HLSColorPalette(colors=[])  # TODO

    @property
    def as_hsv(self) -> 'HSVColorPalette':
        return HSVColorPalette(colors=[])  # TODO

    # Convert hue range to wavelength range in nanometers
    @staticmethod
    def _hue_to_wavelength(hue: int, palette: 'SpectralColorPalette') -> int:
        for color_term in palette.colors:
            hue_range = color_term.hue_approximation
            wavelen_min = color_term.wavelen_nm_min
            wavelen_max = color_term.wavelen_nm_max

            # Check if hue falls within the color's hue range
            if hue_range.min_hue_approximation <= hue <= hue_range.max_hue_approximation:
                # Interpolate the wavelength within this range based on the hue
                hue_span = hue_range.max_hue_approximation - hue_range.min_hue_approximation
                if hue_span == 0:
                    # Handle case for ranges with a single hue (no interpolation needed)
                    return wavelen_min
                # Calculate the interpolated wavelength
                wavelength = wavelen_min + ((hue - hue_range.min_hue_approximation) / hue_span) * (
                        wavelen_max - wavelen_min)
                return int(wavelength)

        # Default return if hue is out of the predefined ranges
        raise ValueError("Hue is out of the defined spectral color palette.")

    def __post_init__(self):
        # Enforce hue values between 0 and 360
        if not (0 <= self.min_hue_approximation <= 360 and 0 <= self.max_hue_approximation <= 360):
            raise ValueError("Hue values must be in the range 0 to 360")


@dataclass
class sRGBAColorPalette:
    colors: List[sRGBAColor]

    @property
    def as_hsv(self) -> 'HSVColorPalette':
        return HSVColorPalette(colors=[c.as_hsv for c in self.colors])

    @property
    def as_hls(self) -> 'HLSColorPalette':
        return HLSColorPalette(colors=[c.as_hls for c in self.colors])


@dataclass
class HSVColorPalette:
    colors: List[HSVColor]

    @property
    def as_rgb(self) -> 'sRGBAColorPalette':
        return sRGBAColorPalette(colors=[c.as_rgb for c in self.colors])

    @property
    def as_hls(self) -> 'HLSColorPalette':
        return HLSColorPalette(colors=[c.as_hls for c in self.colors])


@dataclass
class HLSColorPalette:
    colors: List[HLSColor]

    @property
    def as_rgb(self) -> 'sRGBAColorPalette':
        return sRGBAColorPalette(colors=[c.as_rgb for c in self.colors])

    @property
    def as_hsv(self) -> 'HSVColorPalette':
        return HSVColorPalette(colors=[c.as_hsv for c in self.colors])


@dataclass
class SpectralColorPalette:
    colors: List[SpectralColor]

    @property
    def as_rgb(self) -> 'sRGBAColorPalette':
        return sRGBAColorPalette(colors=[c.as_rgb for c in self.colors])

    @property
    def as_hsv(self) -> 'HSVColorPalette':
        return HSVColorPalette(colors=[c.as_hsv for c in self.colors])

    @property
    def as_hls(self) -> 'HLSColorPalette':
        return HLSColorPalette(colors=[c.as_hls for c in self.colors])


# Ranges taken from https://en.wikipedia.org/wiki/Spectral_color#Spectral_color_terms
NewtonSpectralColorTerms = SpectralColorPalette(colors=[
    SpectralColor(name="Violet", wavelen_nm_min=380, wavelen_nm_max=420,
                  hex_approximation="#7F00FF",
                  hue_approximation=HueRange(min_hue_approximation=249,
                                             max_hue_approximation=250)
                  ),
    SpectralColor(name="Indigo", wavelen_nm_min=430, wavelen_nm_max=440,
                  hex_approximation="#3F00FF",
                  hue_approximation=HueRange(min_hue_approximation=247,
                                             max_hue_approximation=249)),
    SpectralColor(name="Blue", wavelen_nm_min=450, wavelen_nm_max=480,
                  hex_approximation="#1DA2DF",
                  hue_approximation=HueRange(min_hue_approximation=226,
                                             max_hue_approximation=245)),
    SpectralColor(name="Green", wavelen_nm_min=490, wavelen_nm_max=520,
                  hex_approximation="#00FF00",
                  hue_approximation=HueRange(min_hue_approximation=122,
                                             max_hue_approximation=190)),
    SpectralColor(name="Yellow", wavelen_nm_min=530, wavelen_nm_max=570,
                  hex_approximation="#FFFF00",
                  hue_approximation=HueRange(min_hue_approximation=62,
                                             max_hue_approximation=117)),
    SpectralColor(name="Orange", wavelen_nm_min=580, wavelen_nm_max=610,
                  hex_approximation="#FF8800",
                  hue_approximation=HueRange(min_hue_approximation=5,
                                             max_hue_approximation=28)),
    SpectralColor(name="Red", wavelen_nm_min=620, wavelen_nm_max=690,
                  hex_approximation="#FF0000",
                  hue_approximation=HueRange(min_hue_approximation=0,
                                             max_hue_approximation=3))
])
ISCCNBSSpectralColorTerms = SpectralColorPalette(colors=[
    SpectralColor(name="Violet", wavelen_nm_min=380, wavelen_nm_max=430,
                  hex_approximation="#7F00FF",
                  hue_approximation=HueRange(min_hue_approximation=249,
                                             max_hue_approximation=250)),
    SpectralColor(name="Blue", wavelen_nm_min=440, wavelen_nm_max=480,
                  hex_approximation="#3F00FF",
                  hue_approximation=HueRange(min_hue_approximation=226,
                                             max_hue_approximation=247)),
    SpectralColor(name="Blue-Green", wavelen_nm_min=490, wavelen_nm_max=490,
                  hex_approximation="#00FFFF",
                  hue_approximation=HueRange(min_hue_approximation=190,
                                             max_hue_approximation=190)),
    SpectralColor(name="Green", wavelen_nm_min=500, wavelen_nm_max=540,
                  hex_approximation="#00FF00",
                  hue_approximation=HueRange(min_hue_approximation=113,
                                             max_hue_approximation=143)),
    SpectralColor(name="Yellow-Green", wavelen_nm_min=550, wavelen_nm_max=570,
                  hex_approximation="#88FF00",
                  hue_approximation=HueRange(min_hue_approximation=62,
                                             max_hue_approximation=104)),
    SpectralColor(name="Yellow", wavelen_nm_min=580, wavelen_nm_max=580,
                  hex_approximation="#FFFF00",
                  hue_approximation=HueRange(min_hue_approximation=28,
                                             max_hue_approximation=28)),
    SpectralColor(name="Orange", wavelen_nm_min=590, wavelen_nm_max=600,
                  hex_approximation="#FF8800",
                  hue_approximation=HueRange(min_hue_approximation=7,
                                             max_hue_approximation=14)),
    SpectralColor(name="Red", wavelen_nm_min=610, wavelen_nm_max=730,
                  hex_approximation="#FF0000",
                  hue_approximation=HueRange(min_hue_approximation=0,
                                             max_hue_approximation=5))
])
MalacaraSpectralColorTerms = SpectralColorPalette(colors=[
    SpectralColor(name="Violet", wavelen_nm_min=380, wavelen_nm_max=420,
                  hex_approximation="#7F00FF",
                  hue_approximation=HueRange(min_hue_approximation=249,
                                             max_hue_approximation=250)),
    SpectralColor(name="Blue", wavelen_nm_min=430, wavelen_nm_max=490,
                  hex_approximation="#3F00FF",
                  hue_approximation=HueRange(min_hue_approximation=190,
                                             max_hue_approximation=248)),
    SpectralColor(name="Cyan", wavelen_nm_min=500, wavelen_nm_max=510,
                  hex_approximation="#00FFFF",
                  hue_approximation=HueRange(min_hue_approximation=126,
                                             max_hue_approximation=143)),
    SpectralColor(name="Green", wavelen_nm_min=500, wavelen_nm_max=560,
                  hex_approximation="#00FF00",
                  hue_approximation=HueRange(min_hue_approximation=93,
                                             max_hue_approximation=122)),
    SpectralColor(name="Yellow", wavelen_nm_min=570, wavelen_nm_max=570,
                  hex_approximation="#FFFF00",
                  hue_approximation=HueRange(min_hue_approximation=62,
                                             max_hue_approximation=62)),
    SpectralColor(name="Orange", wavelen_nm_min=580, wavelen_nm_max=620,
                  hex_approximation="#FF8800",
                  hue_approximation=HueRange(min_hue_approximation=3,
                                             max_hue_approximation=28)),
    SpectralColor(name="Red", wavelen_nm_min=630, wavelen_nm_max=730,
                  hex_approximation="#FF0000",
                  hue_approximation=HueRange(min_hue_approximation=0,
                                             max_hue_approximation=2))
])
CRCHandbookSpectralColorTerms = SpectralColorPalette(colors=[
    SpectralColor(name="Violet", wavelen_nm_min=380, wavelen_nm_max=440,
                  hex_approximation="#7F00FF",
                  hue_approximation=HueRange(min_hue_approximation=247,
                                             max_hue_approximation=250)),
    SpectralColor(name="Blue", wavelen_nm_min=450, wavelen_nm_max=490,
                  hex_approximation="#3F00FF",
                  hue_approximation=HueRange(min_hue_approximation=190,
                                             max_hue_approximation=245)),
    SpectralColor(name="Green", wavelen_nm_min=500, wavelen_nm_max=560,
                  hex_approximation="#00FF00",
                  hue_approximation=HueRange(min_hue_approximation=93,
                                             max_hue_approximation=143)),
    SpectralColor(name="Yellow", wavelen_nm_min=570, wavelen_nm_max=580,
                  hex_approximation="#FFFF00",
                  hue_approximation=HueRange(min_hue_approximation=28,
                                             max_hue_approximation=62)),
    SpectralColor(name="Orange", wavelen_nm_min=590, wavelen_nm_max=610,
                  hex_approximation="#FF8800",
                  hue_approximation=HueRange(min_hue_approximation=5,
                                             max_hue_approximation=14)),
    SpectralColor(name="Red", wavelen_nm_min=620, wavelen_nm_max=740,
                  hex_approximation="#FF0000",
                  hue_approximation=HueRange(min_hue_approximation=0,
                                             max_hue_approximation=3))
])

IRSpectralColors = SpectralColorPalette(colors=[
    SpectralColor(
        wavelen_nm_min=700,
        wavelen_nm_max=1_000_000,  # Wavelengths can go up to millimeters
        hex_approximation="#000000",  # Black for non-visible
        name="Infrared"
    ),
    SpectralColor(
        wavelen_nm_min=1_000_000,  # 1 mm in nanometers
        wavelen_nm_max=1_000_000_000,  # 1 meter in nanometers
        hex_approximation="#000000",  # Black for non-visible
        name="Microwaves"
    ),
    SpectralColor(
        wavelen_nm_min=1_000_000_000,  # 1 meter
        wavelen_nm_max=100_000_000_000_000,  # 100 km in nanometers
        hex_approximation="#000000",  # Black for non-visible
        name="Radio Waves"
    )
])
UVSpectralColors = SpectralColorPalette(colors=[
    SpectralColor(
        wavelen_nm_min=10,
        wavelen_nm_max=400,
        hex_approximation="#FFFFFF",  # White for invisible light
        name="Ultraviolet"
    ),
    SpectralColor(
        wavelen_nm_min=0.01,
        wavelen_nm_max=10,
        hex_approximation="#FFFFFF",  # White for high energy
        name="X-Rays"
    ),
    SpectralColor(
        wavelen_nm_min=0,
        wavelen_nm_max=0.01,
        hex_approximation="#FFFFFF",  # White for extreme high energy
        name="Gamma Rays"
    )
])
ElectroMagneticSpectrum = SpectralColorPalette(colors=IRSpectralColors.colors +
                                                      ISCCNBSSpectralColorTerms.colors +
                                                      UVSpectralColors.colors)


@dataclass
class ColorTerm:
    name: str
    hue: HueRange
    hex_approximation: Optional[str] = None

    @property
    def as_rgb(self) -> sRGBAColor:
        if self.hex_approximation:
            return sRGBAColor.from_hex_str(self.hex_approximation)
        if self.hue.hex_approximation:
            return sRGBAColor.from_hex_str(self.hue.hex_approximation)
        return self.hue.as_spectral_color.as_rgb


@dataclass
class LanguageColorVocabulary:
    terms: List[ColorTerm]


# Approximate hue ranges for basic colors
EnglishColorTerms = LanguageColorVocabulary(terms=[
    ColorTerm("red", HueRange(0, 30), "#FF0000"),
    ColorTerm("orange", HueRange(30, 60), "#FFA500"),
    ColorTerm("yellow", HueRange(60, 90), "#FFFF00"),
    ColorTerm("green", HueRange(90, 150), "#008000"),
    ColorTerm("cyan", HueRange(150, 180), "#00FFFF"),
    ColorTerm("blue", HueRange(180, 240), "#0000FF"),
    ColorTerm("purple", HueRange(240, 270, "#800080")),
    ColorTerm("magenta", HueRange(270, 300, "#FF00FF")),
    ColorTerm("pink", HueRange(300, 330, "#FFC0CB")),
    ColorTerm("red", HueRange(330, 360, "#FF0000"))
])

# for Typing
Color = Union[sRGBAColor, HSVColor, HLSColor, SpectralColor, ColorTerm]
ColorPalette = Union[sRGBAColorPalette, HSVColorPalette, HLSColorPalette, SpectralColorPalette]


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


def _load_color_json(lang: str) -> Dict[str, str]:
    global _COLOR_DATA
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


def _get_color_adjectives(lang: str) -> Dict[str, List[str]]:
    lang = lang.lower().split("-")[0]
    path = f"{os.path.dirname(__file__)}/res/{lang}/color_descriptors.json"
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _get_object_colors(lang: str) -> Dict[str, str]:
    lang = lang.lower().split("-")[0]
    path = f"{os.path.dirname(__file__)}/res/{lang}/object_colors.json"
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


class FuzzyColor:
    hue_range: HueRange
    approximation: Color

    @property
    def as_hsv(self) -> HSVColor:
        if isinstance(self.approximation, HSVColor):
            return self.approximation
        return self.approximation.as_hsv

    @property
    def as_hls(self) -> HLSColor:
        if isinstance(self.approximation, HLSColor):
            return self.approximation
        return self.approximation.as_hls

    @property
    def as_rgb(self) -> sRGBAColor:
        if isinstance(self.approximation, sRGBAColor):
            return self.approximation
        return self.approximation.as_rgb

    @property
    def as_spectral_color(self) -> SpectralColor:
        if isinstance(self.approximation, SpectralColor):
            return self.approximation
        return self.approximation.as_spectral_color


#################
# TODO - keyword matcher class to encapsulate this
_color_automatons: Dict[str, ahocorasick.Automaton] = {}
_object_automatons: Dict[str, ahocorasick.Automaton] = {}


def _norm(k):
    return k.lower().replace("-", " ").replace("_", " ").strip(" ,.!\n:;")


def _load_color_automaton(lang: str) -> ahocorasick.Automaton:
    global _color_automatons
    if lang in _color_automatons:
        return _color_automatons[lang]
    automaton = ahocorasick.Automaton()
    for hex_str, name in _load_color_json(lang).items():
        automaton.add_word(_norm(name), hex_str)
    automaton.make_automaton()
    _color_automatons[lang] = automaton
    return automaton


def _load_object_automaton(lang: str) -> ahocorasick.Automaton:
    global _object_automatons
    if lang in _object_automatons:
        return _object_automatons[lang]
    automaton = ahocorasick.Automaton()
    for hex_str, name in _get_object_colors(lang).items():
        automaton.add_word(_norm(name), hex_str)
    automaton.make_automaton()
    _object_automatons[lang] = automaton
    return automaton


#################


def color_from_description(description: str, lang: str = "en",
                           strategy: MatchStrategy = MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY,
                           cast_to_palette: bool = False) -> Optional[sRGBAColor]:
    candidates: List[HLSColor] = []
    weights: List[float] = []

    # step 1 - match color db
    color_dict = _load_color_json(lang)
    automaton = _load_color_automaton(lang)
    for idx, hex_str in automaton.iter(description):
        name = color_dict[hex_str]
        weights.append(fuzzy_match(name,
                                   description,
                                   strategy=strategy))
        candidates.append(HLSColor.from_hex_str(hex_str, name=name))
        # print(f"DEBUG: matched color name -> {name}:{hex_str}")

    # Step 2 - match object names
    obj_dict = _get_object_colors(lang)
    automaton = _load_color_automaton(lang)
    for idx, hex_str in automaton.iter(description):
        name = obj_dict[hex_str]
        weights.append(fuzzy_match(name,
                                   description,
                                   strategy=strategy))
        candidates.append(HLSColor.from_hex_str(hex_str, name=name))
        # print(f"DEBUG: matched object name -> {name}:{hex_str}")

    # Step 3 - select base color
    # TODO - add concept of "FuzzyColor" object and allow returning a range of hues instead
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
        raise ValueError("color temperature out of range, only values between 1000 and 4000 supported")

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

    return sRGBAColor(red, green, blue, description=f"{colour_temperature}K")
