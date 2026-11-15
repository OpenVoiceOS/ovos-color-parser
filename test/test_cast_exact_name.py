"""A cast returns the catalogue entry whose name is the whole description.

color_from_description(..., cast_to_palette=True) returns one of the matched
catalogue colours. When one of them has the requested name itself, that entry
is the answer, before any distance to the blend of all matches counts. The
common palette comes first, as in lookup_name, and inside one file the first
entry for a name wins. The expected hex codes are read from the resource file,
not from the parser.
"""
import json
import os
import unittest

import ovos_color_parser
from ovos_color_parser import closest_color, color_from_description, sRGBAColor

COLORS = os.path.join(os.path.dirname(ovos_color_parser.__file__),
                      "res", "en-US", "colors.json")


def first_hex(name):
    with open(COLORS) as f:
        return next(h.upper() for h, n in json.load(f).items() if n == name)


class TestCastExactName(unittest.TestCase):

    def test_names_cast_to_their_own_entry(self):
        for name in ("Airy Blue", "Camel Brown", "Celestial Blue",
                     "Dull Apricot", "Forever Blue", "Gold Medal",
                     "Granada Sky", "Opal Violet", "Parlour Blue",
                     "Prairie Sand", "Scarlet Ibis"):
            with self.subTest(name=name):
                c = color_from_description(name, cast_to_palette=True)
                self.assertEqual(c.hex_str, first_hex(name))

    def test_red_casts_to_red(self):
        c = color_from_description("red", cast_to_palette=True)
        self.assertEqual(c.hex_str, first_hex("Red"))
        self.assertEqual(c.hex_str, "#FF0000")
        self.assertEqual(c.name, "Red")

    def test_closest_color_prefers_the_named_option(self):
        near = sRGBAColor(250, 0, 0, name="Near")
        far = sRGBAColor(0, 0, 255, name="Blue")
        target = sRGBAColor(255, 0, 0)
        self.assertIs(closest_color(target, [near, far]), near)
        self.assertIs(closest_color(target, [near, far], name="blue"), far)
        self.assertIs(closest_color(target, [near, far], name="green"), near)


if __name__ == "__main__":
    unittest.main()
