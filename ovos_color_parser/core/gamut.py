"""Gamut handling: what to do with an *impossible* color.

Some computed colors cannot be shown on an sRGB device: a blackbody at 1500 K, a
pure spectral wavelength, or the result of pushing saturation past the display's
limits. In linear-light coordinates those show up as channels outside ``[0, 1]``.

The old code silently clamped (or worse, constructors raised ``ValueError`` mid-
pipeline). Neither is good: clamping without saying so hides that the request was
unsatisfiable, and throwing turns a recoverable "too bright" into a crash. This
module makes the choice explicit and, crucially, *reports* whether a fit happened
so callers can flag ``out_of_gamut`` instead of guessing.
"""
from enum import Enum
from typing import Tuple

from ovos_color_parser.core.space import LinearRGB


class GamutPolicy(str, Enum):
    """How to resolve a color that falls outside the sRGB gamut.

    - ``CLAMP``: independently clip each channel to ``[0, 1]``. Fast, but can
      shift hue (clipping only red pushes a color towards yellow/magenta).
    - ``MAP``: scale the whole color towards mid-grey until it fits, preserving
      hue at the cost of some chroma. The perceptually kinder default.
    - ``REJECT``: refuse — raise ``ValueError``. For callers that would rather
      handle "no such displayable color" themselves.
    """

    CLAMP = "clamp"
    MAP = "map"
    REJECT = "reject"


_EPS = 1e-9


def in_gamut(lin: LinearRGB) -> bool:
    """True if every color channel already sits within the displayable range."""
    return all(-_EPS <= c <= 1 + _EPS for c in (lin.r, lin.g, lin.b))


def _clamp(lin: LinearRGB) -> LinearRGB:
    clip = lambda c: max(0.0, min(1.0, c))
    return LinearRGB(clip(lin.r), clip(lin.g), clip(lin.b), clip(lin.a))


def _map_towards_grey(lin: LinearRGB) -> LinearRGB:
    """Desaturate towards the color's own luminance until it fits the cube.

    Keeps hue (the direction from grey) fixed and only reduces how far the color
    sits from grey, which is what "bring it into gamut without changing the color
    the user asked for" should mean.
    """
    # relative luminance in linear light (Rec. 709 primaries)
    y = 0.2126 * lin.r + 0.7152 * lin.g + 0.0722 * lin.b
    y = max(0.0, min(1.0, y))
    channels = (lin.r, lin.g, lin.b)
    # smallest scale that pulls every channel back inside [0, 1] around grey `y`
    scale = 1.0
    for c in channels:
        if c > 1 + _EPS:
            scale = min(scale, (1.0 - y) / (c - y)) if c != y else scale
        elif c < -_EPS:
            scale = min(scale, (0.0 - y) / (c - y)) if c != y else scale
    scale = max(0.0, min(1.0, scale))
    mixed = LinearRGB(
        y + (lin.r - y) * scale,
        y + (lin.g - y) * scale,
        y + (lin.b - y) * scale,
        lin.a,
    )
    # tiny float overshoot can remain; finish with a hard clip
    return _clamp(mixed)


def fit_to_gamut(lin: LinearRGB,
                 policy: GamutPolicy = GamutPolicy.CLAMP) -> Tuple[LinearRGB, bool]:
    """Return ``(fitted_color, was_out_of_gamut)``.

    ``was_out_of_gamut`` lets callers surface an honest flag. With
    ``GamutPolicy.REJECT`` an out-of-gamut input raises ``ValueError`` instead of
    returning.
    """
    if in_gamut(lin):
        # still clamp float dust so downstream byte conversion is exact
        return _clamp(lin), False
    if policy == GamutPolicy.REJECT:
        raise ValueError("color is outside the sRGB gamut")
    if policy == GamutPolicy.MAP:
        return _map_towards_grey(lin), True
    return _clamp(lin), True
