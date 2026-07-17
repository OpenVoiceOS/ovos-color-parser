"""The same color words across the supported languages."""
from ovos_color_parser import color_from_description, lookup_name, sRGBAColor

WORDS = {
    "en": "dark red",
    "de": "dunkles rot",
    "fr": "rouge sombre",
    "es": "rojo",
    "pt": "vermelho escuro",
    "it": "rosso",
    "nl": "rood",
    "pl": "czerwony",
    "ru": "красный",
    "cs": "červená",
    "da": "rød",
    "ca": "vermell",
    "eu": "gorria",
    "ar": "أحمر",
    "ro": "roșu",
    "sk": "červená",
    "hr": "crvena",
    "bg": "червено",
    "oc": "roge",
    "an": "royo",
    "ast": "bermeyu",
    "fy": "read",
}

for lang, word in WORDS.items():
    color = color_from_description(word, lang=lang)
    print(f"{lang}: {word!r:20} -> {color.hex_str}")

# name a color in different languages
red = sRGBAColor.from_hex_str("#FF0000")
for lang in ("en", "pt", "ru"):
    print(lang, "->", lookup_name(red, lang=lang))
