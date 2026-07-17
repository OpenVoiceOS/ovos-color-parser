"""Vietnamese (vi-VN) color parsing behavior."""
import unittest

from ovos_color_parser import color_from_description, lookup_name, sRGBAColor


def hue(c):
    return c.as_hls.h


class TestVietnamese(unittest.TestCase):
    def test_basic_colors_resolve_to_expected_hue(self):
        cases = {
            "đỏ": lambda h: h <= 40 or h >= 320,       # red
            "xanh dương": lambda h: 180 <= h <= 280,   # blue
            "xanh lá": lambda h: 80 <= h <= 170,       # green
            "vàng": lambda h: 40 <= h <= 70,           # yellow
            "cam": lambda h: 20 <= h <= 45,            # orange
            "tím": lambda h: 270 <= h <= 320,          # purple
        }
        for word, ok in cases.items():
            with self.subTest(word=word):
                c = color_from_description(word, "vi-VN", fuzzy=False)
                self.assertIsNotNone(c, f"no match for {word}")
                self.assertTrue(ok(hue(c)), f"{word} -> {c.hex_str} h={hue(c)}")

    def test_exact_hex_for_black_and_white(self):
        self.assertEqual(
            color_from_description("đen", "vi-VN", fuzzy=False).hex_str.upper(), "#000000")
        self.assertEqual(
            color_from_description("trắng", "vi-VN", fuzzy=False).hex_str.upper(), "#FFFFFF")

    def test_dark_modifier_lowers_lightness(self):
        base = color_from_description("đỏ", "vi-VN", fuzzy=False)
        dark = color_from_description("đỏ tối", "vi-VN", fuzzy=False)
        self.assertIsNotNone(dark)
        self.assertLess(dark.as_hls.l, base.as_hls.l)

    def test_lookup_name_is_native(self):
        name = lookup_name(sRGBAColor.from_hex_str("#FF0000"), "vi-VN")
        self.assertEqual(name, "đỏ")

    def test_gibberish_returns_none(self):
        self.assertIsNone(color_from_description("qzxwvqq", "vi-VN"))


if __name__ == "__main__":
    unittest.main()
