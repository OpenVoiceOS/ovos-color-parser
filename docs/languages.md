# Language support

Language resources live in `ovos_color_parser/res/<locale>/` as JSON wordlists mapping hex codes
to names. A requested language tag resolves to the best-matching bundled locale: exact
case-insensitive tag first (`pt-br` → `pt-BR`), then any locale sharing the primary language
subtag (`en`, `en-GB` → `en-US`). Unsupported languages make `color_from_description` return
`None` and `palette_from_description` return an empty palette.

## Feature matrix

| Locale | Color name entries | Modifier keywords¹ | Object colors² | Wordlist files |
|---|---|---|---|---|
| ca-ES | 3134 | no | no | 5 |
| cs-CZ | 54 | no | no | 1 |
| da-DK | 14712 | yes | yes | 16 |
| de-DE | 1548 | yes | no | 4 |
| en-US | 14712 | yes | yes | 16 |
| es-ES | 7250 | yes | no | 4 |
| eu-ES | 8011 | yes | no | 5 |
| fr-FR | 14712 | yes | yes | 16 |
| it-IT | ~6900 | yes | no | 5 |
| nl-NL | 101 | no | no | 1 |
| pl-PL | 173 | no | no | 1 |
| pt-BR | ~8300 | yes | yes | 15 |
| ru-RU | 216 | no | no | 1 |

¹ `color_descriptors.json`: saturation/brightness/temperature/opacity adjectives ("dark", "vivid",
"warm", "transparent"). Locales without it still match color names; modifiers are simply ignored.

² `object_colors.json`: prototypical object colors ("carrot", "banana", "blood").

Every public function accepts every language: locales lacking a resource degrade gracefully
instead of raising.

## Wordlist sources

Depending on the locale, wordlists include translations of: basic colors, the xkcd color survey,
crayola crayons, RAL classic/design/effect/plastics, Pantone, ISCC-NBS, .NET named colors,
traditional Japanese colors and Wikipedia's list of colors.

## Adding a language

1. Create `ovos_color_parser/res/<lang>-<REGION>/colors.json` with `{"#RRGGBB": "name", ...}`.
2. Optionally add more wordlist files (any `*.json` with the same shape).
3. Optionally add `color_descriptors.json` (copy the key structure from `en-US`) and
   `object_colors.json`.
4. Add anchor words for the language to `test/test_languages.py`.
