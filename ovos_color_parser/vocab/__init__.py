"""Vocabularies: the named-color data and how it is loaded.

A "namespace" here is one palette — one ``res/<lang>/<name>.json`` file such as
``webcolors``, ``crayola`` or ``RAL_classic``. Treating each as a first-class,
addressable thing (rather than melting them all into one flat lookup) is what lets
a caller ask for "the RAL name" or restrict matching to basic web colors, and what
makes ``lookup_name`` deterministic instead of dependent on directory iteration
order.
"""
from ovos_color_parser.vocab.loader import (
    load_locale_palettes,
    load_shared_palettes,
    load_palettes,
    palette_names,
    iter_color_dicts,
)

__all__ = [
    "load_locale_palettes",
    "load_shared_palettes",
    "load_palettes",
    "palette_names",
    "iter_color_dicts",
]
