# -*- coding: utf-8 -*-
"""Hebrew color-term resolution against the bundled ``he`` wordlists."""
import unittest

from ovos_color_parser import color_from_description


def _is_reddish(color):
    h = color.as_hls.h
    return h <= 40 or h >= 320


def _is_bluish(color):
    return 180 <= color.as_hls.h <= 280


def _is_greenish(color):
    return 80 <= color.as_hls.h <= 170


class TestHebrew(unittest.TestCase):
    def test_basic_colors_resolve(self):
        for word, check in (('אדום', _is_reddish),
                            ('כחול', _is_bluish),
                            ('ירוק', _is_greenish)):
            with self.subTest(word=word):
                c = color_from_description(word, "he")
                self.assertIsNotNone(c, f"no match for {word!r}")
                self.assertTrue(check(c), f"{word!r} -> {c.hex_str} wrong hue")

    def test_darkness_modifier_lowers_lightness(self):
        base = color_from_description('אדום', "he")
        dark = color_from_description('אדום כהה', "he")
        self.assertIsNotNone(dark)
        self.assertLess(dark.as_hls.l, base.as_hls.l)

    def test_gibberish_returns_none(self):
        self.assertIsNone(color_from_description("qzxwvqq", "he"))


if __name__ == "__main__":
    unittest.main()
