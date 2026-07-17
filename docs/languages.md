# Language support

Language resources live in `ovos_color_parser/res/<locale>/` as JSON wordlists mapping hex codes
to names. A requested language tag resolves to the best-matching bundled locale: exact
case-insensitive tag first (`pt-br` → `pt-BR`), then any locale sharing the primary language
subtag (`en`, `en-GB` → `en-US`). Unsupported languages make `color_from_description` return
`None` and `palette_from_description` return an empty palette.

## Feature matrix

| Locale | Language | Color name entries | Modifier keywords¹ | Object colors² |
|---|---|---|---|---|
| an-ES | Aragonese | 14 | no | no |
| ar-SA | Arabic | 72 | yes | yes |
| ast-ES | Asturian | 17 | no | no |
| bg-BG | Bulgarian | 32 | no | no |
| ca-ES | Catalan | 3134 | no | no |
| cs-CZ | Czech | 54 | no | no |
| da-DK | Danish | 14712 | yes | yes |
| de-DE | German | 1548 | yes | no |
| en-US | English | 14712 | yes | yes |
| es-ES | Spanish | 7250 | yes | no |
| eu-ES | Basque | 8011 | yes | no |
| fr-FR | French | 14712 | yes | yes |
| fy-NL | West Frisian | 15 | no | no |
| hr-HR | Croatian | 30 | no | no |
| it-IT | Italian | 6820 | yes | no |
| kab-DZ | Kabyle | 12 | no | no |
| nl-NL | Dutch | 101 | no | no |
| oc-FR | Occitan | 23 | yes | no |
| pl-PL | Polish | 173 | no | no |
| pt-BR | Portuguese | 8238 | yes | yes |
| ro-RO | Romanian | 101 | yes | no |
| ru-RU | Russian | 216 | no | no |
| sk-SK | Slovak | 35 | no | no |

23 locales. `color_from_description` accepts any BCP-47 tag and resolves it to the closest
bundled locale.

¹ `color_descriptors.json`: saturation/brightness/temperature/opacity adjectives ("dark", "vivid",
"warm", "transparent"). Locales without it still match color names; modifiers are simply ignored.

² `object_colors.json`: prototypical object colors ("carrot", "banana", "blood").

Every public function accepts every language: locales lacking a resource degrade gracefully
instead of raising.

The largest wordlists (Danish, English, French) share the full ~14.7k-entry set covering web
colors, the xkcd survey, crayola, RAL, Pantone and more. Smaller locales ship curated basic and
traditional color terms; they resolve common names accurately but will snap unusual descriptions
to a nearer basic color.

## Arabic notes

Arabic ships four color wordlists plus object colors and modifier keywords. `colors.json`
holds Modern Standard Arabic (fuṣḥā) basic and extended shades — teal (`أزرق مخضر`),
cyan (`سيان`), coral (`مرجاني`), lavender (`خزامي`), khaki (`كاكي`), mustard (`خردلي`),
mint (`نعناعي`), amber (`كهرماني`), burgundy (`عنابي`) and more. `dialectal_colors.json`
adds colloquial terms spanning the major dialect areas: Egyptian (`بمبي` pink, `لبني`
light blue, `بترولي` petrol), Levantine/Shami (`روز` rose, `جوزي` walnut brown),
Gulf/Khaleeji (`قهوائي` coffee-brown, `عسلي` honey, `سكري` sugar-cream), Iraqi
(`طوبي` brick-red, `جكليتي` chocolate-brown) and Maghrebi/Darija (`بلو` blue). Multiple
names map to a shared hex, so synonyms and dialectal variants resolve to the same color.

Because normalization does not strip Arabic diacritics or unify letter forms,
`orthographic_variants.json` ships the spellings a user might type differently: the
hamza-less initial alef (`احمر` for `أحمر`), alef-maqṣūra for final yāʾ (`بنفسجى`), and
an explicitly vowelled form (`أَحْمَر`). `object_colors.json` maps prototypical objects to
their color, including dialectal object words (`طماطم` vs Levantine `بندورة` for tomato,
`موز` banana-yellow, `سماء` sky-blue, `دم` blood-red, `ذهب` gold). `color_descriptors.json`
covers Standard and dialectal modifiers for brightness, saturation, temperature and
opacity (dark `غامق`/`غانق`, light `فاتح`/`فاقع`).

A locale-level Arabic normalizer (folding hamza forms, tāʾ marbūṭa and tashkeel at match
time) would let a single canonical spelling cover every written variant and is a natural
follow-up; the shipped variant entries keep matching robust without touching shared code.

## Kabyle notes

Kabyle colexifies blue and green: `azegzaw` covers both and is mapped to both hexes ("grue").
`anili` (indigo), `azenǧǧari` (sky blue) and `amuri` (navy) name specific blues; `adal`,
`azemmuri` (olive) and `aqesli` (light green) are attested greens. Only attested terms ship;
modifier keywords are omitted for lack of sources.

## Wordlist sources

Depending on the locale, wordlists include translations of: basic colors, the xkcd color survey,
crayola crayons, RAL classic/design/effect/plastics, Pantone, ISCC-NBS, .NET named colors,
traditional Japanese colors and Wikipedia's list of colors.

## Adding a language

See [extending.md](extending.md#adding-a-language) for the step-by-step guide to adding a locale,
shipping custom wordlists, and enabling modifier keywords.
