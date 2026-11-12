"""A catalogue colour keeps its hex code through the HLS and HSV models.

color_from_description(..., cast_to_palette=True) returns one of the
catalogue colours ("do not invent colors"), so its hex code is the one the
catalogue file lists. Each 8-bit sRGB colour converts to HLS or HSV and back
to the same 8-bit values, as long as the hue keeps its fraction of a degree.
The expected hex codes are read from the resource files, not from the parser.
"""
import json
import os
import unittest

import ovos_color_parser
from ovos_color_parser import color_from_description
from ovos_color_parser.models import HLSColor, HSVColor

RES = os.path.join(os.path.dirname(ovos_color_parser.__file__), "res", "en-US")


def six_digits(hex_str):
    """'#abc' is short for '#aabbcc' (CSS Color 4, section 5.2)."""
    hex_str = hex_str.upper()
    if len(hex_str) == 4:
        return "#" + "".join(ch * 2 for ch in hex_str[1:])
    return hex_str


def catalogue(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


class TestPaletteHexRoundTrip(unittest.TestCase):

    def test_issue_colors_cast_to_their_catalogue_hex(self):
        colors = catalogue("colors.json")
        for name, hex_str in (("Earthworm", "#C62D42"), ("OVOS red", "#FF1A1A")):
            with self.subTest(name=name):
                self.assertEqual(colors[hex_str], name)
                c = color_from_description(name, cast_to_palette=True)
                self.assertEqual(c.hex_str, hex_str)

    def test_every_catalogue_hex_survives_hls_and_hsv(self):
        hexes = {six_digits(h) for h in catalogue("colors.json")}
        self.assertGreater(len(hexes), 5000)
        for model in (HLSColor, HSVColor):
            with self.subTest(model=model.__name__):
                changed = [h for h in sorted(hexes)
                           if model.from_hex_str(h).hex_str != h]
                self.assertEqual(changed[:5], [],
                                 f"{len(changed)} of {len(hexes)} change")


if __name__ == "__main__":
    unittest.main()
