"""Color science primitives shared by the whole library.

``core`` owns the *numbers*: how a color is represented internally, how it is
converted between spaces, and how out-of-gamut results are handled. It knows
nothing about language, vocabularies or parsing.

- :mod:`ovos_color_parser.core.space` — the canonical working space
  (linear-light sRGB) and a single ``convert`` entry point. All blending and
  adjustment math happens here so it stays perceptually honest instead of being
  smeared across gamma-encoded HLS.
- :mod:`ovos_color_parser.core.gamut` — deciding what to do when a computed
  color falls outside the sRGB gamut (``clamp`` / ``map`` / ``reject``).
"""
from ovos_color_parser.core.distance import srgb8_to_lab, srgb8_distance, delta_e_cie2000
from ovos_color_parser.core.gamut import GamutPolicy, in_gamut, fit_to_gamut
from ovos_color_parser.core.space import (
    LinearRGB,
    srgb8_to_linear,
    linear_to_srgb8,
    blend_linear,
)

__all__ = [
    "LinearRGB",
    "srgb8_to_linear",
    "linear_to_srgb8",
    "blend_linear",
    "GamutPolicy",
    "in_gamut",
    "fit_to_gamut",
    "srgb8_to_lab",
    "srgb8_distance",
    "delta_e_cie2000",
]
