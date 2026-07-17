"""Cached, namespace-aware loading of the color vocabularies.

The old loader re-read and re-parsed every JSON file on every call, and only ever
looked inside the requested locale directory — so the shared, language-neutral
``res/webcolors.json`` never loaded at all. This module fixes both: files are
parsed once and cached, palettes keep their file-stem name, and the shared
palettes are available to every locale as a base namespace.

Palettes are returned in a deterministic priority order so that "what is this
color called" resolves the same way every run (common names before obscure
industrial catalogs), instead of depending on ``os.listdir`` order.
"""
import json
import os
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, OrderedDict as OrderedDictT
from collections import OrderedDict

_RES_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "res")

# Common, human-facing palettes first; industrial / niche catalogs last. Any
# palette not named here is appended alphabetically, keeping the order total and
# stable. ``object_colors`` sits at the end: it maps things (carrot, sky) to
# colors and is a poor source of canonical color *names*.
_PREFERRED_ORDER = (
    "colors",
    "webcolors",
    "99colors",
    "crayola",
    "xkcd_colors",
    "wikipedia_color_list",
    "ISCC-NBS",
    "japan_colors",
    "dot-net-colors",
    "pantome_colors",
    "RAL_classic",
    "RAL_design",
    "RAL_effect",
    "RAL_plastics_p1",
    "RAL_plastics_p2",
    "object_colors",
)

# Palettes that are never part of the name/color vocabulary.
_NON_PALETTE_FILES = frozenset({"color_descriptors.json"})


def _ordered(names: Iterable[str]) -> List[str]:
    names = set(names)
    head = [n for n in _PREFERRED_ORDER if n in names]
    tail = sorted(names - set(head))
    return head + tail


@lru_cache(maxsize=None)
def _read_json(path: str) -> Dict[str, str]:
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=None)
def _resolve_lang_dir(lang: str) -> Optional[str]:
    """Requested language tag -> existing ``res/<locale>`` directory, or None.

    Prefers an exact case-insensitive locale match, then any directory sharing the
    primary subtag (``en-GB`` -> ``en-US``).
    """
    if not lang:
        return None
    available = [d for d in os.listdir(_RES_ROOT)
                 if os.path.isdir(os.path.join(_RES_ROOT, d))]
    by_lower = {d.lower(): d for d in available}
    requested = lang.lower().replace("_", "-")
    if requested in by_lower:
        return os.path.join(_RES_ROOT, by_lower[requested])
    primary = requested.split("-")[0]
    if primary in by_lower:
        return os.path.join(_RES_ROOT, by_lower[primary])
    for d in sorted(available):
        if d.lower().split("-")[0] == primary:
            return os.path.join(_RES_ROOT, d)
    return None


@lru_cache(maxsize=None)
def load_shared_palettes() -> OrderedDictT:
    """Language-neutral palettes shipped at the ``res/`` root (e.g. ``webcolors``).

    Available as a base namespace for every locale.
    """
    palettes = {}
    for fname in os.listdir(_RES_ROOT):
        if not fname.endswith(".json") or fname in _NON_PALETTE_FILES:
            continue
        palettes[fname[:-5]] = _read_json(os.path.join(_RES_ROOT, fname))
    return OrderedDict((n, palettes[n]) for n in _ordered(palettes))


@lru_cache(maxsize=None)
def load_locale_palettes(lang: str) -> OrderedDictT:
    """Palettes from the locale directory only, in priority order.

    Object-name vocabularies and descriptor lists are excluded — this is the
    color-name namespace set used for matching within a single language.
    """
    res_dir = _resolve_lang_dir(lang)
    if not res_dir:
        return OrderedDict()
    palettes = {}
    for fname in os.listdir(res_dir):
        if not fname.endswith(".json") or fname in _NON_PALETTE_FILES:
            continue
        palettes[fname[:-5]] = _read_json(os.path.join(res_dir, fname))
    return OrderedDict((n, palettes[n]) for n in _ordered(palettes))


def load_palettes(lang: str = "en", include_shared: bool = True) -> OrderedDictT:
    """All palettes visible for ``lang``: locale palettes first, then shared ones.

    Locale palettes take priority so a localized name wins over the neutral
    ``webcolors`` name for the same hex.
    """
    merged = OrderedDict(load_locale_palettes(lang))
    if include_shared:
        for name, palette in load_shared_palettes().items():
            merged.setdefault(name, palette)
    return merged


def palette_names(lang: str = "en", include_shared: bool = True) -> List[str]:
    """Names of the palettes/namespaces available for ``lang``."""
    return list(load_palettes(lang, include_shared).keys())


def iter_color_dicts(lang: str) -> Iterable[Dict[str, str]]:
    """Back-compat helper: yield each locale color dict (no object/descriptor
    lists), matching the old ``_load_color_json`` contract but cached."""
    return list(load_locale_palettes(lang).values())
