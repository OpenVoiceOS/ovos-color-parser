"""NER / entity extraction: pull color mentions out of free text.

No OVOS, no ML model, no network -- just the bundled color wordlists and a
sliding window. Given a sentence, find the color phrases and resolve each to a
structured color (name, hex, rgb). Useful for tagging product descriptions,
design briefs, support tickets or any free text that mentions colors.

    pip install ovos-color-parser
    python ner_colors.py

Strategy: color-name matching is substring based, so `fuzzy=False` keeps filler
words ("the", "fence") from matching. We find maximal runs of consecutive words
that each resolve on their own, then pull in immediately-adjacent modifier words
("dark", "warm", "light") using the library's own bundled descriptor wordlists,
and finally resolve the whole span together so the modifiers are applied.
"""
from dataclasses import dataclass
from typing import List, Set

from ovos_color_parser import color_from_description
from ovos_color_parser.matching import _get_color_adjectives


def _modifier_words(lang: str) -> Set[str]:
    """Flatten the bundled saturation/brightness/temperature/opacity adjectives."""
    words: Set[str] = set()
    for group in _get_color_adjectives(lang).values():
        words.update(w.lower() for w in group)
    return words


def _resolves(word: str, lang: str) -> bool:
    return color_from_description(word, lang=lang, fuzzy=False) is not None


@dataclass
class ColorEntity:
    text: str        # the exact span from the sentence
    start: int       # word index where the span starts
    end: int         # word index (exclusive) where it ends
    hex: str         # resolved hex code
    rgb: tuple       # resolved (r, g, b)


def extract_colors(text: str, lang: str = "en") -> List[ColorEntity]:
    words = [w.strip(".,;:!?\"'") for w in text.split()]
    modifiers = _modifier_words(lang)
    colorish = [_resolves(w, lang) for w in words]

    entities: List[ColorEntity] = []
    i = 0
    while i < len(words):
        if not colorish[i]:
            i += 1
            continue
        # grow a run of consecutive resolving words
        j = i
        while j + 1 < len(words) and colorish[j + 1]:
            j += 1
        start, end = i, j + 1
        # absorb a leading modifier ("dark forest green", "warm mustard")
        while start > 0 and words[start - 1].lower() in modifiers:
            start -= 1
        phrase = " ".join(words[start:end])
        color = color_from_description(phrase, lang=lang, fuzzy=False)
        if color is not None:
            entities.append(ColorEntity(phrase, start, end,
                                        color.hex_str, (color.r, color.g, color.b)))
        i = end
    return entities


if __name__ == "__main__":
    text = ("Paint the fence dark forest green and the gate warm mustard "
            "yellow, but keep the door navy blue.")
    print(f"input: {text}\n")
    for ent in extract_colors(text):
        print(f"  [{ent.start}:{ent.end}] {ent.text!r:22} -> {ent.hex} rgb{ent.rgb}")

    print()
    # works in other languages too - just pass lang=
    text_pt = "quero a parede verde escuro e o carro azul"
    print(f"input (pt): {text_pt}\n")
    for ent in extract_colors(text_pt, lang="pt"):
        print(f"  [{ent.start}:{ent.end}] {ent.text!r:22} -> {ent.hex} rgb{ent.rgb}")
