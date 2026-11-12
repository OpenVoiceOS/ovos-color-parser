"""Order the translated ``colors.json`` files so that a basic colour word casts to its basic colour.

Some locales translate ``res/en-US/colors.json`` entry by entry. A cast with
``cast_to_palette=True`` returns the first entry whose name matches the whole
description, so the entry order of a translated file decides which colour a word
such as "verde" casts to. For each translated file, this script:

1. puts the entries in the order of ``res/en-US/colors.json``. Entries that have
   no en-US counterpart come after them, in their current order.
2. finds the word the file uses for each basic colour term: the translation it
   gives most often to the entries that en-US names with the bare term, the
   first such entry breaking a tie. The first of those entries moves to the head
   of the entries that carry that word. The translation of a modified name
   ("Artichoke green") can be the bare word ("Grøn"); this step stops such an
   entry from going in front of the entry for the basic term itself ("Green").

The basic colour terms of English are the eleven of Berlin and Kay (1969),
"Basic Color Terms: Their Universality and Evolution", with both spellings of grey.

A file is a translation when it holds at least 90% of the en-US hex codes.
Run from the repository root: ``python scripts/order_locale_colors.py``
"""
import json
import os
import sys

from ovos_color_parser.matching import _norm

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ovos_color_parser", "res")
BASIC_TERMS = {"white", "black", "red", "green", "yellow", "blue", "brown",
               "purple", "pink", "orange", "grey"}
MIN_SHARED = 0.9


def ordered_keys(colors: dict, reference: dict) -> list:
    keys = [h for h in reference if h in colors]
    keys += [h for h in colors if h not in reference]
    by_term = {}
    for h in keys:
        term = _norm(reference.get(h, ""))
        term = "grey" if term == "gray" else term
        if term in BASIC_TERMS:
            by_term.setdefault(term, []).append(h)
    heads = {}
    for entries in by_term.values():
        names = [_norm(colors[h]) for h in entries]
        word = max(names, key=lambda n: (names.count(n), -names.index(n)))
        heads[word] = entries[names.index(word)]
    result = {}
    for h in keys:
        result.setdefault(heads.get(_norm(colors[h]), h))
        result.setdefault(h)
    return list(result)


def main() -> int:
    reference = json.load(open(os.path.join(RES, "en-US", "colors.json"), encoding="utf-8"))
    for lang in sorted(os.listdir(RES)):
        path = os.path.join(RES, lang, "colors.json")
        if lang == "en-US" or not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        colors = json.loads(text)
        if sum(h in colors for h in reference) < MIN_SHARED * len(reference):
            continue
        keys = ordered_keys(colors, reference)
        assert sorted(keys) == sorted(colors), lang
        unmatched = sum(h not in reference for h in colors)
        print(f"{lang}: {len(colors)} entries in, {len(keys)} out, {unmatched} with no en-US entry")
        if keys == list(colors):
            continue
        indent = len(text.splitlines()[1]) - len(text.splitlines()[1].lstrip())
        out = json.dumps({h: colors[h] for h in keys}, indent=indent, ensure_ascii=False)
        with open(path, "w", encoding="utf-8") as f:
            f.write(out + ("\n" if text.endswith("\n") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
