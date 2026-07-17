"""Behavior tests for the Finnish (fi) color vocabulary."""
import unittest

from ovos_color_parser import color_from_description, sRGBAColor


def is_reddish(c: sRGBAColor) -> bool:
    h = c.as_hls.h
    return h <= 40 or h >= 320


def is_bluish(c: sRGBAColor) -> bool:
    return 180 <= c.as_hls.h <= 280


def is_greenish(c: sRGBAColor) -> bool:
    return 80 <= c.as_hls.h <= 170


class TestFiColors(unittest.TestCase):
    LANG = "fi-FI"

    def _c(self, text):
        return color_from_description(text, self.LANG)

    def test_basic_colors_resolve(self):
        for word in ['punainen', 'sininen', 'vihreä', 'keltainen', 'musta', 'valkoinen', 'oranssi', 'violetti', 'ruskea', 'vaaleanpunainen']:
            with self.subTest(word=word):
                self.assertIsNotNone(self._c(word), f"no match for {word!r}")

    def test_red_is_reddish(self):
        self.assertTrue(is_reddish(self._c("punainen")))

    def test_blue_is_bluish(self):
        self.assertTrue(is_bluish(self._c("sininen")))

    def test_green_is_greenish(self):
        self.assertTrue(is_greenish(self._c("vihreä")))

    def test_dark_modifier_lowers_lightness(self):
        base = self._c("punainen")
        dark = self._c("tumma punainen")
        self.assertIsNotNone(dark)
        self.assertLess(dark.as_hls.l, base.as_hls.l)

    def test_light_modifier_shifts(self):
        base = self._c("vihreä")
        light = self._c("kirkas vihreä")
        self.assertIsNotNone(light)
        self.assertGreater(light.as_hls.l, base.as_hls.l)

    def test_gibberish_returns_none(self):
        self.assertIsNone(self._c("qzxwvqq"))


if __name__ == "__main__":
    unittest.main()
