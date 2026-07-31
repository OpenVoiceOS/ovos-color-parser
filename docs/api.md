# API reference

All public names are importable from `ovos_color_parser`.

## Parsing

### `color_from_description(description, lang="en", strategy=..., cast_to_palette=False, fuzzy=True, gamut=GamutPolicy.CLAMP) -> Optional[sRGBAColor]`

Parse a natural-language color description.

- Matches the description against the language's bundled color wordlists and object-color list.
- Averages all matched colors in linear light, weighted by match confidence (see `average_colors`).
- Applies modifier keywords from the language's descriptor list: saturation ("vivid", "muted"),
  brightness ("light", "dark"), temperature ("warm", "cool") and opacity ("opaque", "transparent" —
  applied to the alpha channel).
- `cast_to_palette=True` returns the matched candidate closest to the averaged color instead of the
  average itself, so the result is always a known, named color.
- `gamut` chooses how a computed color that leaves the sRGB gamut is resolved:
  `GamutPolicy.CLAMP` (per-channel, default), `GamutPolicy.MAP` (hue-preserving), or
  `GamutPolicy.REJECT`.
- Returns `None` when nothing matches (including unsupported languages).

```python
>>> color_from_description("dark red").hex_str
'#371013'
>>> color_from_description("qzxwv") is None
True
```

### `palette_from_description(description, lang="en", strategy=...) -> sRGBAColorPalette`

Return every color matched by the description as a palette (empty palette when nothing matches).

### `lookup_name(color, lang="en", namespace=None, nearest=False) -> str`

Return the name of `color` from the known vocabularies.

- `namespace` restricts the lookup to a single palette (e.g. `"webcolors"`, `"crayola"`,
  `"RAL_classic"`); see `palette_names(lang)` for the available namespaces.
- Resolution is deterministic (common palettes before niche catalogs).
- `nearest=True` falls back to the perceptually closest named color in scope instead of raising.
- Raises `ValueError` for an unnamed color (exact miss with `nearest=False`) or an unknown namespace.

### Namespaces (`ovos_color_parser.vocab`)

`palette_names(lang)`, `load_palettes(lang)`, `load_locale_palettes(lang)` and
`load_shared_palettes()` expose the color palettes as named, cached namespaces. The
language-neutral `webcolors` palette is available as a base namespace for every locale.

### `ColorMatcher(lang="en", color_palettes=None, object_colors=None)`

Instantiable name spotter. Use the class methods for the cached global path, or construct an
instance to match against a fixed language or injected custom vocabularies
(`match_colors(description, fuzzy=False)`, `match_objects(description)`).

## Color models (`ovos_color_parser.models`)

| Class | Fields | Notes |
|---|---|---|
| `sRGBAColor` | `r, g, b` (0-255), `a` (0-255), `name`, `description` | `.hex_str`, `.from_hex_str()` (3- or 6-digit), `.as_hls`, `.as_hsv`, `.as_spectral_color` |
| `HSVColor` | `h` (0-360), `s, v` (0-1) | `.as_rgb`, `.as_hls`, `.hex_str`, `.from_hex_str()` |
| `HLSColor` | `h` (0-360), `l, s` (0-1) | `.as_rgb`, `.as_hsv`, `.hex_str`, `.from_hex_str()` |
| `SpectralColor` | `wavelen_nm_min`, `wavelen_nm_max` | wavelength (nm) based; `.as_rgb` etc. |
| `HueRange` | `min_hue_approximation`, `max_hue_approximation` | hue interval; `.as_spectral_color` |
| `ColorTerm` | `name`, `hue`, `hex_approximation` | a language's color word; either field derives the other |

All constructors validate ranges and raise `ValueError` on out-of-range values.
Palettes (`sRGBAColorPalette`, `HSVColorPalette`, `HLSColorPalette`, `SpectralColorPalette`) are
lists of colors with `.as_rgb` / `.as_hls` / `.as_hsv` conversions.

Bundled vocabularies: `EnglishColorTerms`, `NewtonSpectralColorTerms`, `ISCCNBSSpectralColorTerms`.

## Utilities

### `color_distance(color_a, color_b) -> float`

Perceptual distance (CIEDE2000 over CIE L\*a\*b\*). Smaller is more similar.

### `closest_color(color, color_opts) -> Color`

Return the option with the smallest `color_distance` to `color`.

### `average_colors(colors, weights=None) -> HLSColor`

Weighted average in HLS space with a circular mean for hue (350° and 10° average to 0°, not 180°).
Raises `ValueError` for an empty list or mismatched weights.

### `convert_K_to_RGB(colour_temperature) -> sRGBAColor`

Black-body color temperature (1000-40000 K) to RGB.

### `get_contrasting_black_or_white(hex_code) -> sRGBAColor`

Black or white, whichever contrasts best with the given background color (YIQ heuristic).

### `is_hex_code_valid(hex_code) -> bool`

`True` for 3- or 6-digit hex codes, with or without a leading `#`.

### `rgb_to_cmyk(r, g, b) -> (c, m, y, k)` / `cmyk_to_rgb(c, m, y, k) -> (r, g, b)`

RGB (0-255) to CMYK (0-100) and back.

## Matching internals (`ovos_color_parser.matching`)

`ColorMatcher` caches one Aho-Corasick automaton per language for exact substring matching and
falls back to fuzzy token matching. Language resources resolve by best locale match: an exact
case-insensitive tag first (`pt-br` → `pt-BR`), then any locale sharing the primary subtag
(`en-GB` → `en-US`).

---
[← Color, language and color spaces](color-theory.md) · [Home](../README.md) · [Language support →](languages.md)
