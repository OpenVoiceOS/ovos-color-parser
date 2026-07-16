# OVOS Color Parser

Parse natural-language color descriptions into color objects, and name colors, in 13 languages.

```python
from ovos_color_parser import color_from_description

c = color_from_description("dark red", lang="en")
print(c.hex_str)   # "#371013"
print(c.as_hls)    # HLSColor(h=352, l=0.145..., s=0.522..., ...)
```

Designed for voice interfaces:

- "change the lamp color to moss green"
- "make it darker"
- "a warmer white"

## Installation

```bash
pip install ovos-color-parser
```

## Features

- **Color extraction** — `color_from_description("light blue", lang="fr")` matches bundled color
  wordlists (web colors, xkcd survey, crayola, RAL, Pantone, ISCC-NBS, traditional Japanese colors, ...)
  and object colors ("carrot", "banana"), then applies modifiers such as *light/dark*, *vivid/muted*,
  *warm/cool* and *transparent/opaque*.
- **Color naming** — `lookup_name(color, lang)` returns the name of a known color.
- **Color models** — `sRGBAColor`, `HLSColor`, `HSVColor` and `SpectralColor` (wavelength) dataclasses
  with conversions, stable hex round-trips and validation.
- **Utilities** — perceptual color distance (CIECAM02 deltaE), weighted color averaging with circular
  hue mean, Kelvin color temperature to RGB, CMYK conversion, contrasting black/white text color and
  hex validation.

## Supported languages

Basque, Catalan, Czech, Danish, Dutch, English, French, German, Italian, Polish, Portuguese, Russian
and Spanish. Any BCP-47 tag resolves to the closest bundled locale (for example `en-GB` → `en-US`).
The per-language feature matrix is in [docs/languages.md](docs/languages.md).

## Usage notes

Color names are ambiguous — the same name can map to several hex values across wordlists. When
several entries match, the parser averages them, weighted by match confidence. To force a known,
named color from the matched candidates instead:

```python
color = color_from_description("red", lang="en", cast_to_palette=True)
print(color.name)  # a named wordlist color, e.g. "Fire engine red"
```

When nothing matches, `color_from_description` returns `None`.

Descriptions of [impossible colors](https://en.wikipedia.org/wiki/Impossible_color)
("reddish green") still produce an output — the parser averages whatever it matches, which may not
be meaningful.

Runnable scripts live in [examples/](examples/) and the full API reference in
[docs/api.md](docs/api.md).

## Related projects

- [ovos-number-parser](https://github.com/OpenVoiceOS/ovos-number-parser) — numbers
- [ovos-date-parser](https://github.com/OpenVoiceOS/ovos-date-parser) — dates and times
- [ovos-lang-parser](https://github.com/OVOSHatchery/ovos-lang-parser) — languages

## Credits

Color wordlists include data derived from the xkcd color survey, crayola, RAL, Pantone, ISCC-NBS,
traditional Japanese colors and Wikipedia color lists. Spectral color terms follow
[Wikipedia's spectral color tables](https://en.wikipedia.org/wiki/Spectral_color#Spectral_color_terms).
