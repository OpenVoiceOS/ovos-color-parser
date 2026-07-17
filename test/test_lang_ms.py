"""Focused checks for the Malay (ms-MY) color wordlist."""
import unittest

from ovos_color_parser import color_from_description, sRGBAColor


def _h(c: sRGBAColor):
    return c.as_hls.h


class TestMalay(unittest.TestCase):
    lang = "ms-MY"

    def test_red_is_reddish(self):
        c = color_from_description("merah", self.lang)
        self.assertIsNotNone(c)
        h = _h(c)
        self.assertTrue(h <= 40 or h >= 320, f"'merah' -> {c.hex_str} not reddish")

    def test_blue_is_bluish(self):
        c = color_from_description("biru", self.lang)
        self.assertIsNotNone(c)
        self.assertTrue(180 <= _h(c) <= 280, f"'biru' -> {c.hex_str} not bluish")

    def test_green_is_greenish(self):
        c = color_from_description("hijau", self.lang)
        self.assertIsNotNone(c)
        self.assertTrue(80 <= _h(c) <= 170, f"'hijau' -> {c.hex_str} not greenish")

    def test_yellow_is_yellowish(self):
        c = color_from_description("kuning", self.lang)
        self.assertIsNotNone(c)
        self.assertTrue(40 <= _h(c) <= 75, f"'kuning' -> {c.hex_str} not yellowish")

    def test_dark_modifier_darkens(self):
        base = color_from_description("merah", self.lang)
        darker = color_from_description("merah gelap", self.lang)
        self.assertIsNotNone(darker)
        self.assertLess(darker.as_hls.l, base.as_hls.l,
                        "gelap modifier should reduce lightness")

    def test_gibberish_returns_none(self):
        self.assertIsNone(color_from_description("qzxwvqq", self.lang))


if __name__ == "__main__":
    unittest.main()
