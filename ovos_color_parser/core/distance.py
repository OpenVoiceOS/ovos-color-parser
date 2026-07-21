"""Perceptual color distance in pure Python.

Nearest-named-color matching needs a distance that tracks how *different two
colors look*, not their raw RGB gap. The standard answer is to convert to CIE Lab
and use the CIEDE2000 difference formula. Doing it here — on top of the linear
conversions already in :mod:`ovos_color_parser.core.space` — keeps the library
free of a heavyweight color-science dependency (and the numpy it drags in) for
what is ultimately a few hundred lines of arithmetic.
"""
import math
from typing import Tuple

from ovos_color_parser.core.space import LinearRGB, srgb8_to_linear

# D65 reference white, 2° observer.
_XN, _YN, _ZN = 95.047, 100.0, 108.883


def linear_to_xyz(lin: LinearRGB) -> Tuple[float, float, float]:
    """linear-light sRGB -> CIE XYZ (D65), scaled to a 0-100 Y range."""
    r, g, b = lin.r, lin.g, lin.b
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) * 100
    y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) * 100
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) * 100
    return x, y, z


def _f(t: float) -> float:
    delta = 6.0 / 29.0
    if t > delta ** 3:
        return t ** (1.0 / 3.0)
    return t / (3 * delta ** 2) + 4.0 / 29.0


def xyz_to_lab(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """CIE XYZ (D65) -> CIE L*a*b*."""
    fx, fy, fz = _f(x / _XN), _f(y / _YN), _f(z / _ZN)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def srgb8_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    return xyz_to_lab(*linear_to_xyz(srgb8_to_linear(r, g, b)))


def delta_e_cie2000(lab1: Tuple[float, float, float],
                    lab2: Tuple[float, float, float]) -> float:
    """CIEDE2000 color difference between two L*a*b* colors.

    ~0 means indistinguishable, ~1-2 is a just-noticeable difference, and pure
    complementary colors sit in the tens. Reference: Sharma, Wu & Dalal (2005).
    """
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2

    avg_lp = (l1 + l2) / 2.0
    c1 = math.hypot(a1, b1)
    c2 = math.hypot(a2, b2)
    avg_c = (c1 + c2) / 2.0

    g = 0.5 * (1 - math.sqrt(avg_c ** 7 / (avg_c ** 7 + 25 ** 7))) if avg_c else 0.0
    a1p = (1 + g) * a1
    a2p = (1 + g) * a2
    c1p = math.hypot(a1p, b1)
    c2p = math.hypot(a2p, b2)
    avg_cp = (c1p + c2p) / 2.0

    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360

    dlp = l2 - l1
    dcp = c2p - c1p

    if c1p * c2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif h2p - h1p > 180:
        dhp = h2p - h1p - 360
    else:
        dhp = h2p - h1p + 360
    dHp = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dhp) / 2.0)

    if c1p * c2p == 0:
        avg_hp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        avg_hp = (h1p + h2p) / 2.0
    elif h1p + h2p < 360:
        avg_hp = (h1p + h2p + 360) / 2.0
    else:
        avg_hp = (h1p + h2p - 360) / 2.0

    t = (1
         - 0.17 * math.cos(math.radians(avg_hp - 30))
         + 0.24 * math.cos(math.radians(2 * avg_hp))
         + 0.32 * math.cos(math.radians(3 * avg_hp + 6))
         - 0.20 * math.cos(math.radians(4 * avg_hp - 63)))

    d_ro = 30 * math.exp(-(((avg_hp - 275) / 25) ** 2))
    rc = 2 * math.sqrt(avg_cp ** 7 / (avg_cp ** 7 + 25 ** 7)) if avg_cp else 0.0
    sl = 1 + (0.015 * (avg_lp - 50) ** 2) / math.sqrt(20 + (avg_lp - 50) ** 2)
    sc = 1 + 0.045 * avg_cp
    sh = 1 + 0.015 * avg_cp * t
    rt = -math.sin(math.radians(2 * d_ro)) * rc

    return math.sqrt(
        (dlp / sl) ** 2
        + (dcp / sc) ** 2
        + (dHp / sh) ** 2
        + rt * (dcp / sc) * (dHp / sh)
    )


def srgb8_distance(rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
    """Perceptual CIEDE2000 distance between two 8-bit sRGB colors."""
    return delta_e_cie2000(srgb8_to_lab(*rgb1), srgb8_to_lab(*rgb2))
