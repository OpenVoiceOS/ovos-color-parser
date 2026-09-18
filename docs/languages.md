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
| az-AZ | Azerbaijani | 43 | yes | no |
| bg-BG | Bulgarian | 32 | no | no |
| ca-ES | Catalan | 3421 | no | no |
| cs-CZ | Czech | 54 | no | no |
| da-DK | Danish | 14712 | yes | yes |
| de-DE | German | 1548 | yes | no |
| el-GR | Greek | 32 | yes | no |
| en-US | English | 14712 | yes | yes |
| es-ES | Spanish | 7250 | yes | no |
| et-EE | Estonian | 87 | yes | no |
| eu-ES | Basque | 8011 | yes | no |
| fa-IR | Persian | 46 | yes | no |
| fi-FI | Finnish | 94 | yes | no |
| fr-FR | French | 14712 | yes | yes |
| fy-NL | West Frisian | 15 | no | no |
| gl-ES | Galician | 32 | yes | no |
| he-IL | Hebrew | 44 | yes | no |
| hi-IN | Hindi | 36 | yes | no |
| hr-HR | Croatian | 30 | no | no |
| hu-HU | Hungarian | 31 | yes | no |
| id-ID | Indonesian | 44 | yes | no |
| it-IT | Italian | 6820 | yes | no |
| ja-JP | Japanese | 40 | yes | no |
| kab-DZ | Kabyle | 12 | no | no |
| ko-KR | Korean | 39 | yes | no |
| ms-MY | Malay | 40 | yes | no |
| mwl-PT | Mirandese | 15 | yes | no |
| nb-NO | Norwegian Bokmål | 93 | yes | no |
| nl-NL | Dutch | 101 | no | no |
| nn-NO | Norwegian Nynorsk | 93 | yes | no |
| oc-FR | Occitan | 23 | yes | no |
| pl-PL | Polish | 173 | no | no |
| pt-BR | Portuguese | 8238 | yes | yes |
| ro-RO | Romanian | 101 | yes | no |
| ru-RU | Russian | 216 | no | no |
| sk-SK | Slovak | 35 | no | no |
| sl-SI | Slovenian | 28 | yes | no |
| sv-SE | Swedish | 94 | yes | no |
| tr-TR | Turkish | 50 | yes | no |
| uk-UA | Ukrainian | 29 | yes | no |
| vi-VN | Vietnamese | 38 | yes | no |
| zh-CN | Chinese | 44 | yes | no |

45 locales. `color_from_description` accepts any BCP-47 tag and resolves it to the closest
bundled locale.

¹ `color_descriptors.json`: saturation/brightness/temperature/opacity adjectives ("dark", "vivid",
"warm", "transparent"). Locales without it still match color names. Modifiers are simply ignored.

² `object_colors.json`: prototypical object colors ("carrot", "banana", "blood").

Every public function accepts every language: locales lacking a resource degrade gracefully
instead of raising.

The largest wordlists (Danish, English, French) share the full ~14.7k-entry set covering web
colors, the xkcd survey, crayola, RAL, Pantone and more. Smaller locales ship curated basic and
traditional color terms. They resolve common names accurately but will snap unusual descriptions
to a nearer basic color.

## Arabic notes

Arabic ships four color wordlists plus object colors and modifier keywords. `colors.json`
holds Modern Standard Arabic (fuṣḥā) basic and extended shades: teal (`أزرق مخضر`),
cyan (`سيان`), coral (`مرجاني`), lavender (`خزامي`), khaki (`كاكي`), mustard (`خردلي`),
mint (`نعناعي`), amber (`كهرماني`), burgundy (`عنابي`) and more.

`dialectal_colors.json` adds colloquial terms spanning the major dialect areas: Egyptian
(`بمبي` pink, `لبني` light blue, `بترولي` petrol), Levantine/Shami (`روز` rose, `جوزي` walnut
brown), Gulf/Khaleeji (`قهوائي` coffee-brown, `عسلي` honey, `سكري` sugar-cream), Iraqi
(`طوبي` brick-red, `جكليتي` chocolate-brown) and Maghrebi/Darija (`بلو` blue). Multiple
names map to a shared hex, so synonyms and dialectal variants resolve to the same color.

Arabic tashkeel (the optional short-vowel and other diacritic marks) are removed during
normalization, so a word matches whether it is written bare or fully vowelled. `أحمر` and
`أَحْمَر` resolve to the same color, and the same holds for dialectal names and modifier
phrases.

Letter-form differences are not diacritics and are handled as data.
`orthographic_variants.json` ships the spellings a user might type differently: the
hamza-less initial alef (`احمر` for `أحمر`) and alef-maqṣūra for final yāʾ (`بنفسجى`).
`object_colors.json` maps prototypical objects to their color, including dialectal object
words (`طماطم` vs Levantine `بندورة` for tomato, `موز` banana-yellow, `سماء` sky-blue,
`دم` blood-red, `ذهب` gold). `color_descriptors.json` covers Standard and dialectal
modifiers for brightness, saturation, temperature and opacity (dark `غامق`/`غانق`, light
`فاتح`/`فاقع`).

Very short names, such as the two-letter `دم` (blood), match only as whole words, never
fuzzily. An unrelated word that merely contains them, such as `قدم` (foot) or `بنيان`
(building), is not misread as a color.

## Kabyle notes

Kabyle colexifies blue and green: `azegzaw` covers both and is mapped to both hexes ("grue").
`anili` (indigo), `azenǧǧari` (sky blue) and `amuri` (navy) name specific blues. `adal`,
`azemmuri` (olive) and `aqesli` (light green) are attested greens. Only attested terms ship.
Modifier keywords are omitted for lack of sources.

## Wordlist sources

Depending on the locale, wordlists include translations of: basic colors, the xkcd color survey,
crayola crayons, RAL classic/design/effect/plastics, Pantone, ISCC-NBS, .NET named colors,
traditional Japanese colors and Wikipedia's list of colors.

## Adding a language

See [extending.md](extending.md#adding-a-language) for the step-by-step guide to adding a locale,
shipping custom wordlists, and enabling modifier keywords.

---
[← API reference](api.md) · [Home](../README.md) · [Extending →](extending.md)
