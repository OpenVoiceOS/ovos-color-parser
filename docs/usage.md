# Usage guide

## Parsing a color from text

`color_from_description` matches color names and object names in a phrase, blends the matches, and
applies any modifier keywords it finds.

```python
from ovos_color_parser import color_from_description

for phrase in ["bright vibrant green", "pale pink", "muted warm gray", "dark cool blue"]:
    c = color_from_description(phrase)
    print(f"{phrase:20} -> {c.hex_str}  {c.name}")
```

Names are matched on word boundaries, so "red" is found in "dark red" but not inside "shredded", and
modifier words behave the same way ("light" applies in "light blue" but not in "flighty"). More
specific names carry more weight in the blend: "moss green" pulls the result toward moss, not toward a
generic green.

An unrecognised phrase, or a language with no bundled locale, yields `None`.

```python
color_from_description("qzxwv")          # None
color_from_description("red", lang="zz") # None
```

## Ambiguity and named colors

The same name can refer to several hex values across wordlists. By default the parser blends every
match in linear light:

```python
color_from_description("red").hex_str    # "#B84D54"
```

Pass `cast_to_palette=True` to snap the result to the closest matched named color instead of the
blend, so the output is always a real, named entry:

```python
c = color_from_description("red", cast_to_palette=True)
print(c.name, c.hex_str)                 # "Dusty Red" "#..."
```

## Modifiers

Modifier keywords adjust the matched color. They fall into four groups, each with a "high" and "low"
direction and a stronger "very" variant:

| Group | Effect | Example words |
|---|---|---|
| Saturation | chroma up/down | vivid, rich / muted, washed-out |
| Brightness | lighter/darker | bright, light / dim, dark |
| Temperature | warmer/cooler | warm, fiery / cool, icy |
| Opacity | alpha up/down | opaque, solid / sheer, transparent |

```python
base = color_from_description("blue")
darker = color_from_description("dark blue")
lighter = color_from_description("light blue")
print(darker.as_hls.l < base.as_hls.l)   # True
print(lighter.as_hls.l > base.as_hls.l)  # True
```

The keyword lists live in each locale's `color_descriptors.json`. A locale without that file still
matches color names; its modifiers are simply not applied. See
[Color description semantics](keywords.md) for the full mapping.

## Color namespaces

Each wordlist is a named namespace — `webcolors`, `crayola`, `RAL_classic`, `xkcd_colors` and so on.

```python
from ovos_color_parser import lookup_name, palette_names, sRGBAColor

palette_names("en")                      # ['colors', '99colors', 'crayola', ...]

red = sRGBAColor.from_hex_str("#FF0000")
lookup_name(red, "en")                                   # "Red"
lookup_name(red, "en", namespace="crayola", nearest=True)  # "Permanent Geranium Lake"
```

`lookup_name` resolves in a fixed priority (common palettes before niche catalogs) so a color always
resolves to the same name. Restrict it to one namespace with `namespace=`, and allow the perceptually
nearest match with `nearest=True`. Without `nearest`, a color that is not in scope raises
`ValueError`.

## Impossible and out-of-gamut colors

Some phrases describe colors that cannot exist. `"reddish green"` and `"yellowish blue"` sit on
opposite ends of an opponent channel; the visual system cancels them rather than perceiving a single
color. The parser still returns something — it blends whatever names it matches:

```python
color_from_description("reddish green")  # a muted result, not a meaningful color
```

A computed color can also fall outside the sRGB gamut, for example after strong warmth or brightness
adjustments. The `gamut` argument decides how that is resolved:

```python
from ovos_color_parser import color_from_description, GamutPolicy

color_from_description("warm bright red", gamut=GamutPolicy.CLAMP)   # clip each channel (default)
color_from_description("warm bright red", gamut=GamutPolicy.MAP)     # desaturate toward grey, keep hue
color_from_description("warm bright red", gamut=GamutPolicy.REJECT)  # raise on out-of-gamut
```

Wavelength colors outside human vision are flagged rather than faked:

```python
from ovos_color_parser.models import IRSpectralColors
IRSpectralColors.colors[0].is_visible    # False
```

## Comparing colors

`color_distance` is the perceptual CIEDE2000 difference — smaller is more similar.

```python
from ovos_color_parser import color_distance, color_from_description

color_distance(color_from_description("green"), color_from_description("yellow"))  # ~44
color_distance(color_from_description("green"), color_from_description("purple"))  # ~63
```

`closest_color` picks the nearest option from a palette:

```python
from ovos_color_parser import sRGBAColor, sRGBAColorPalette, closest_color

palette = sRGBAColorPalette(colors=[
    sRGBAColor(0, 128, 128, name="Teal"),
    sRGBAColor(64, 224, 208, name="Turquoise"),
    sRGBAColor(0, 63, 255, name="Cerulean"),
])
closest_color(sRGBAColor(0, 0, 255, name="Blue"), palette.colors).name  # "Cerulean"
```

## Isolated and custom matching

`ColorMatcher` can be instantiated for a single language, or with injected vocabularies for a closed
set of colors:

```python
from ovos_color_parser import ColorMatcher

matcher = ColorMatcher("en")
matcher.match_colors("moss green")

custom = ColorMatcher("xx", color_palettes=[{"#FF0000": "rood", "#0000FF": "blauw"}])
custom.match_colors("maak het rood")     # matches "rood"
```
