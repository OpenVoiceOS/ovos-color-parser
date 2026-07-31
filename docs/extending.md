# Extending and integrating

This page covers using the library outside a voice assistant, tuning the matcher, adding a
language, and shipping your own wordlists.

## Exact vs. fuzzy matching

`color_from_description(..., fuzzy=True)` (the default) is forgiving: it tolerates typos and loose
phrasing, which is what a spoken or typed request usually needs. The trade-off is that short filler
words can incidentally match a color name.

For entity extraction over arbitrary prose, pass `fuzzy=False`. Matching then requires a color name
to appear as a substring, so words like "the", "door" or "keep" resolve to `None` and only genuine
color phrases survive. [examples/ner_colors.py](../examples/ner_colors.py) builds a span extractor
on top of this.

```python
from ovos_color_parser import color_from_description

color_from_description("keep the door shut", fuzzy=True)   # may match something
color_from_description("keep", fuzzy=False)                # None
color_from_description("navy blue", fuzzy=False).hex_str   # '#0B32B4'
```

## Reusing the bundled modifier wordlists

Locales with a `color_descriptors.json` ship saturation/brightness/temperature/opacity adjectives.
You can read them to detect modifier words in your own text processing:

```python
from ovos_color_parser.matching import _get_color_adjectives

adjectives = _get_color_adjectives("en")   # {"low_brightness": ["dark", ...], ...}
modifiers = {w.lower() for group in adjectives.values() for w in group}
"dark" in modifiers   # True
```

## Standalone vs. OVOS

Nothing in the package imports OVOS, so the same call works in a script or a skill — only the input
plumbing differs.

```python
# standalone
from ovos_color_parser import color_from_description
hex_str = color_from_description("moss green", lang="en").hex_str

# inside a skill handler
def handle_set_color(self, message):
    color = color_from_description(message.data["utterance"], lang=self.lang)
    if color:
        self.set_lamp(color.hex_str)
```

Because `lang` is just a BCP-47 tag, wiring the library into any framework is a matter of passing
that framework's active language through.

## Custom wordlists at runtime

Wordlists are plain `{"#RRGGBB": "name"}` JSON. To parse against your own palette, add a locale
directory (see below) or reuse the models directly:

```python
from ovos_color_parser import sRGBAColor, closest_color

brand = {"#0A66C2": "Brand Blue", "#F5A623": "Brand Amber", "#2E2E2E": "Brand Ink"}
palette = [sRGBAColor.from_hex_str(h, name=n) for h, n in brand.items()]

picked = closest_color(sRGBAColor.from_hex_str("#0B5FB0"), palette)
print(picked.name)   # 'Brand Blue'
```

## Adding a language

1. Create `ovos_color_parser/res/<lang>-<REGION>/colors.json` with `{"#RRGGBB": "name", ...}`.
2. Optionally add more wordlist files (any `*.json` with the same shape) — they are all merged.
3. Optionally add `color_descriptors.json` (copy the key structure from `en-US`) to enable
   light/dark/vivid/warm/transparent modifiers, and `object_colors.json` for prototypical object
   colors ("carrot", "banana").
4. Add anchor words for the language to `test/test_languages.py`.

A requested tag resolves to the closest bundled locale: exact case-insensitive tag first, then any
locale sharing the primary language subtag. See [languages.md](languages.md) for the current
locale list and per-language vocabulary notes.

---
[← Language support](languages.md) · [Home](../README.md)
