"""Canonical working space + conversions.

The public color types (``sRGBAColor``/``HSVColor``/``HLSColor``/...) are *views*
onto real light. When you need to combine or nudge colors you cannot do it in the
view coordinates and expect physically sensible results: averaging two gamma-
encoded sRGB values, or two HLS lightnesses, darkens and muddies the mix because
the encoding is non-linear.

So the library keeps one canonical space — **linear-light sRGB**, channels as
floats that may temporarily leave ``[0, 1]`` — and does every blend/adjust there.
``convert`` is the single funnel in and out; adding a new space means adding one
pair of functions, not an accessor on every existing type (the old design grew a
quadratic web of ``as_*`` properties).
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# sRGB transfer function constants (IEC 61966-2-1).
_GAMMA_THRESHOLD_ENCODED = 0.04045
_GAMMA_THRESHOLD_LINEAR = 0.0031308


def _decode_channel(c: float) -> float:
    """gamma-encoded sRGB [0,1] -> linear-light [0,1]."""
    if c <= _GAMMA_THRESHOLD_ENCODED:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _encode_channel(c: float) -> float:
    """linear-light [0,1] -> gamma-encoded sRGB [0,1]. Values outside [0,1] pass
    through the branch that keeps them monotonic so gamut handling can see them."""
    if c <= _GAMMA_THRESHOLD_LINEAR:
        return c * 12.92
    return 1.055 * (max(c, 0.0) ** (1 / 2.4)) - 0.055


@dataclass
class LinearRGB:
    """A color in linear-light sRGB. Channels are floats and MAY fall outside
    ``[0, 1]`` — that is exactly the signal that a computed color is out of gamut,
    which :mod:`ovos_color_parser.core.gamut` then resolves."""

    r: float
    g: float
    b: float
    a: float = 1.0

    def blend(self, other: "LinearRGB", t: float) -> "LinearRGB":
        """Linear interpolation towards ``other`` by fraction ``t`` in [0,1]."""
        return LinearRGB(
            self.r + (other.r - self.r) * t,
            self.g + (other.g - self.g) * t,
            self.b + (other.b - self.b) * t,
            self.a + (other.a - self.a) * t,
        )


def srgb8_to_linear(r: int, g: int, b: int, a: int = 255) -> LinearRGB:
    """8-bit gamma-encoded sRGB (0-255) -> linear-light."""
    return LinearRGB(
        _decode_channel(r / 255),
        _decode_channel(g / 255),
        _decode_channel(b / 255),
        a / 255,
    )


def linear_to_srgb8(lin: LinearRGB) -> Tuple[int, int, int, int]:
    """linear-light -> 8-bit gamma-encoded sRGB (0-255), rounded and clamped to
    the byte range. This clamp is a *representation* clamp (a channel cannot be
    stored as -3 or 300); true gamut decisions belong to ``core.gamut`` and should
    happen on the ``LinearRGB`` value before this call."""
    def _to_byte(c: float) -> int:
        return max(0, min(255, round(_encode_channel(c) * 255)))

    return (
        _to_byte(lin.r),
        _to_byte(lin.g),
        _to_byte(lin.b),
        max(0, min(255, round(lin.a * 255))),
    )


def blend_linear(colors: Sequence[LinearRGB],
                 weights: Optional[Sequence[float]] = None) -> LinearRGB:
    """Weighted average of colors in linear light.

    This is the physically correct way to mix colors: it is what happens when you
    overlap light sources, and it avoids the darkening artefact of averaging in a
    gamma-encoded or HLS space. Weights need not sum to 1; they are normalised.
    """
    colors = list(colors)
    if not colors:
        raise ValueError("colors must be a non-empty list")
    if weights is None:
        weights = [1.0] * len(colors)
    else:
        weights = list(weights)
        if len(weights) != len(colors):
            raise ValueError("weights must have the same length as colors")
    total = sum(weights)
    if total == 0:
        # degenerate: treat as an unweighted mean rather than dividing by zero
        weights = [1.0] * len(colors)
        total = float(len(colors))
    return LinearRGB(
        sum(c.r * w for c, w in zip(colors, weights)) / total,
        sum(c.g * w for c, w in zip(colors, weights)) / total,
        sum(c.b * w for c, w in zip(colors, weights)) / total,
        sum(c.a * w for c, w in zip(colors, weights)) / total,
    )
