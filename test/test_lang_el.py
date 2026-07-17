# -*- coding: utf-8 -*-
"""Behaviour test for the Greek (el-GR) colour locale."""
import unittest

from ovos_color_parser import color_from_description


class TestGreekColors(unittest.TestCase):
    LANG = "el"

    def _hue(self, word):
        c = color_from_description(word, self.LANG)
        self.assertIsNotNone(c, f"no match for {word!r}")
        return c

    def test_red_is_reddish(self):
        h = self._hue('κόκκινο').as_hls.h
        self.assertTrue(h <= 40 or h >= 320, f"red hue {h} not reddish")

    def test_green_is_greenish(self):
        h = self._hue('πράσινο').as_hls.h
        self.assertTrue(80 <= h <= 170, f"green hue {h} not greenish")

    def test_blue_is_bluish(self):
        h = self._hue('μπλε').as_hls.h
        self.assertTrue(180 <= h <= 280, f"blue hue {h} not bluish")

    def test_dark_modifier_lowers_brightness(self):
        plain = self._hue('κόκκινο')
        dark = color_from_description("σκούρο κόκκινο", self.LANG)
        self.assertIsNotNone(dark)
        self.assertLess(dark.as_hls.l, plain.as_hls.l,
                        "dark modifier did not darken the colour")

    def test_gibberish_returns_none(self):
        self.assertIsNone(color_from_description("qzxwvqq", self.LANG))


if __name__ == "__main__":
    unittest.main()
