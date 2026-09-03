# OVOS Color Parser

Turn natural-language color descriptions into color objects, and color objects back into
names, in 23 languages. It is pure Python, needs no network access and no ML model. It uses
bundled wordlists and color math.

```python
from ovos_color_parser import color_from_description

c = color_from_description("dark red", lang="en")
print(c.hex_str)   # "#3B1315"
print(c.as_hls)    # HLSColor(h=357, l=0.153..., s=0.513..., ...)
```

It ships as part of the OpenVoiceOS voice stack, but it is a standalone library first: nothing
here imports OVOS. It is equally useful for NER over free text, describing colors for
text-to-speech, and mapping color descriptions to hex for UI theming or LED control.

## Installation

```bash
pip install ovos-color-parser
# or, with uv:
uv pip install ovos-color-parser
```

## 30-second quickstart

```python
from ovos_color_parser import color_from_description, lookup_name, sRGBAColor

# text -> color
c = color_from_description("warm mustard yellow", lang="en")
print(c.hex_str, (c.r, c.g, c.b))     # #FFDA3E (255, 218, 62)

# color -> text (any supported language)
print(lookup_name(sRGBAColor.from_hex_str("#1E90FF"), lang="pt"))  # "Azul furtivo"

# nothing matches -> None
print(color_from_description("qzxwv", lang="en"))  # None
```

## Features

- **Color extraction**: `color_from_description("light blue", lang="fr")` matches bundled color
  wordlists (web colors, xkcd survey, crayola, RAL, Pantone, ISCC-NBS, traditional Japanese colors, ...)
  and object colors ("carrot", "banana"), then applies modifiers such as *light/dark*, *vivid/muted*,
  *warm/cool* and *transparent/opaque*. Names are matched on word boundaries and weighted by
  specificity, so "moss green" outweighs a bare "green" and "green" is never matched inside "evergreen".

- **Color naming and namespaces**: `lookup_name(color, lang)` returns a color's name. Every wordlist
  is an addressable namespace. Ask for the name in a specific palette (`namespace="RAL_classic"`),
  or fall back to the perceptually nearest named color (`nearest=True`).

- **Color models**: `sRGBAColor`, `HLSColor`, `HSVColor` and `SpectralColor` (wavelength) dataclasses
  with conversions, stable hex round-trips and validation. `SpectralColor.is_visible` separates real
  colors from infrared, ultraviolet and beyond.

- **Gamut handling**: choose how a computed color that leaves the sRGB gamut is resolved. Clamp
  each channel, map towards grey while preserving hue, or reject it (`gamut=GamutPolicy.MAP`).

- **Utilities**: perceptual color distance (CIEDE2000), linear-light color averaging, Kelvin color
  temperature to RGB, CMYK conversion, contrasting black/white text color and hex validation.

## Use it anywhere

Every snippet below is plain Python. Run `pip install ovos-color-parser`, then run it. Each has a
matching runnable script in [examples/](examples/).

### Extract colors from free text (NER)

Pull color references out of a sentence and resolve each to a structured color, with no ML model
and no network access. This is useful for tagging product copy, design briefs or support tickets.

```python
from ovos_color_parser import color_from_description

text = "Paint the fence dark forest green and the door navy blue"
for phrase in ("dark forest green", "navy blue"):
    c = color_from_description(phrase, lang="en", fuzzy=False)
    print(phrase, "->", c.hex_str)   # dark forest green -> #162914 ; navy blue -> #0F43BE
```

Full sliding-window extractor with span offsets: [examples/ner_colors.py](examples/ner_colors.py).

### Describe a color out loud (TTS-adjacent)

Go the other way. A raw RGB/hex value from a color picker, sensor or smart bulb becomes a
speakable name.

```python
from ovos_color_parser import sRGBAColor, lookup_name

c = sRGBAColor.from_hex_str("#2E8B57")
print(f"The color is {lookup_name(c, lang='en').lower()}.")   # "The color is sea green."
```

See [examples/describe_rgb.py](examples/describe_rgb.py).

### Map descriptions to hex for UI, theming and LEDs

```python
from ovos_color_parser import color_from_description

theme = {var: color_from_description(desc, lang="en").hex_str.lower()
         for var, desc in {"--accent": "vivid teal", "--bg": "very dark blue"}.items()}
print(theme)   # {'--accent': '#38bfc4', '--bg': '#121a34'}
```

CSS variables, NeoPixel/WLED tuples and Kelvin white-balance in
[examples/ui_theming.py](examples/ui_theming.py).

### In an OVOS skill vs. standalone

The same call powers a voice intent and a plain script. Only the surrounding code differs:

```python
# standalone color utility
from ovos_color_parser import color_from_description
hex_str = color_from_description("moss green", lang="en").hex_str

# inside an OVOS skill handler
def handle_set_color(self, message):
    utterance = message.data["utterance"]
    color = color_from_description(utterance, lang=self.lang)
    if color:
        self.set_lamp(color.hex_str)
```

## Supported languages

23 locales: Aragonese, Arabic, Asturian, Basque, Bulgarian, Catalan, Croatian, Czech, Danish,
Dutch, English, French, German, Italian, Kabyle, Occitan, Polish, Portuguese, Romanian, Russian,
Slovak, Spanish and West Frisian. Any BCP-47 tag resolves to the closest bundled locale, for
example `en-GB` resolves to `en-US`. The per-language feature matrix, with entry counts and
modifier and object support, is in [docs/languages.md](docs/languages.md).

## Documentation

- [Usage guide](docs/usage.md): extraction, impossible colors, comparing colors

- [Color description semantics](docs/keywords.md): how hue, saturation, brightness,
  temperature and opacity keywords map to color math

- [Color, language and color spaces](docs/color-theory.md): how languages carve up
  color space, and the color models used

- [API reference](docs/api.md)

- [Language support](docs/languages.md): the 23 locales and their per-language vocabulary

- [Extending](docs/extending.md): custom wordlists, adding a language, integration patterns

### Runnable examples

- [parse_colors.py](examples/parse_colors.py): descriptions to colors, modifiers, `cast_to_palette`

- [describe_rgb.py](examples/describe_rgb.py): RGB/hex to a spoken color name (TTS)

- [ner_colors.py](examples/ner_colors.py): extract color spans from free text (NER)

- [ui_theming.py](examples/ui_theming.py): descriptions to CSS/LED/Kelvin values

- [multilingual.py](examples/multilingual.py): the same colors across all languages

- [color_math.py](examples/color_math.py): models, conversions, distance and averaging

## Usage notes

Color names are ambiguous. The same name can map to several hex values across wordlists. When
several entries match, the parser blends them in linear light, weighted by match specificity. To
force a known, named color from the matched candidates instead:

```python
color = color_from_description("red", lang="en", cast_to_palette=True)
print(color.name)  # a named wordlist color, e.g. "Dusty Red"
```

When nothing matches, `color_from_description` returns `None`.

Descriptions of [impossible colors](https://en.wikipedia.org/wiki/Impossible_color)
("reddish green") still produce an output. The parser blends whatever it matches, and the result
may not be meaningful.

Runnable scripts live in [examples/](examples/) and the full API reference in
[docs/api.md](docs/api.md).

## Related projects

- [ovos-number-parser](https://github.com/OpenVoiceOS/ovos-number-parser): numbers

- [ovos-date-parser](https://github.com/OpenVoiceOS/ovos-date-parser): dates and times

- [ovos-lang-parser](https://github.com/OVOSHatchery/ovos-lang-parser): languages

## Credits

Color wordlists include data derived from the xkcd color survey, crayola, RAL, Pantone, ISCC-NBS,
traditional Japanese colors and Wikipedia color lists. Spectral color terms follow
[Wikipedia's spectral color tables](https://en.wikipedia.org/wiki/Spectral_color#Spectral_color_terms).
