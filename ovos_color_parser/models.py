from colorsys import rgb_to_hsv, hsv_to_rgb, hls_to_rgb, rgb_to_hls
from dataclasses import dataclass
from typing import List, Optional, Union


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


@dataclass
class ColorTerm:
    name: str
    hue: Optional[HueRange] = None
    hex_approximation: Optional[str] = None

    @property
    def as_rgb(self) -> sRGBAColor:
        if self.hex_approximation:
            return sRGBAColor.from_hex_str(self.hex_approximation)
        if self.hue.hex_approximation:
            return sRGBAColor.from_hex_str(self.hue.hex_approximation)
        return self.hue.as_spectral_color.as_rgb

    def __post_init__(self):
        if not self.hue and self.hex_approximation:
            hls = HLSColor.from_hex_str(self.hex_approximation)
            self.hue = HueRange(hls.h - 15, hls.h + 15, name=self.name, hex_approximation=self.hex_approximation)
        elif not self.hex_approximation and self.hue:
            self.hex_approximation = self.as_rgb.hex_str


@dataclass
class LanguageColorVocabulary:
    terms: List[ColorTerm]


# for Typing
Color = Union[sRGBAColor, HSVColor, HLSColor, SpectralColor, ColorTerm]
ColorPalette = Union[sRGBAColorPalette, HSVColorPalette, HLSColorPalette, SpectralColorPalette]

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

# using xcolor and terms defined in https://www.nature.com/articles/s41599-022-01045-3
OtjihereroColorTerms = LanguageColorVocabulary(
    terms=[
        ColorTerm("grine", hex_approximation="#76a479"),
        ColorTerm("vapa", hex_approximation="#bdc5ce"),
        ColorTerm("ngara", hex_approximation="#c7b665"),
        ColorTerm("dumbu", hex_approximation="#c1ae6c"),
        ColorTerm("burou", hex_approximation="#6d79ac"),
        ColorTerm("zoozu", hex_approximation="#42303b"),
        ColorTerm("zorondu", hex_approximation="#42303b"),
        ColorTerm("vinde", hex_approximation="#6d4953"),
        ColorTerm("pinge", hex_approximation="#cd779b"),
        ColorTerm("ranje", hex_approximation="#d0767b"),
        ColorTerm("serandu", hex_approximation="#bc5a72"),
    ]
)
