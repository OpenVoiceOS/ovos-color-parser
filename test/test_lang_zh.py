"""Mandarin Chinese (zh-CN) color parsing behavior."""
import unittest

from ovos_color_parser import color_from_description, lookup_name, sRGBAColor


def hue(c):
    return c.as_hls.h


class TestChinese(unittest.TestCase):
    def test_basic_colors_resolve_to_expected_hue(self):
        # word -> predicate on hue (degrees)
        cases = {
            "红色": lambda h: h <= 40 or h >= 320,   # red
            "蓝色": lambda h: 180 <= h <= 280,        # blue
            "绿色": lambda h: 80 <= h <= 170,         # green
            "黄色": lambda h: 40 <= h <= 70,          # yellow
            "橙色": lambda h: 20 <= h <= 45,          # orange
            "紫色": lambda h: 270 <= h <= 320,        # purple
        }
        for word, ok in cases.items():
            with self.subTest(word=word):
                c = color_from_description(word, "zh-CN", fuzzy=False)
                self.assertIsNotNone(c, f"no match for {word}")
                self.assertTrue(ok(hue(c)), f"{word} -> {c.hex_str} h={hue(c)}")

    def test_exact_hex_for_black_and_white(self):
        self.assertEqual(
            color_from_description("黑色", "zh-CN", fuzzy=False).hex_str.upper(), "#000000")
        self.assertEqual(
            color_from_description("白色", "zh-CN", fuzzy=False).hex_str.upper(), "#FFFFFF")

    def test_dark_modifier_lowers_lightness(self):
        base = color_from_description("红色", "zh-CN", fuzzy=False)
        dark = color_from_description("深 红色", "zh-CN", fuzzy=False)
        self.assertIsNotNone(dark)
        self.assertLess(dark.as_hls.l, base.as_hls.l)

    def test_lookup_name_is_native(self):
        name = lookup_name(sRGBAColor.from_hex_str("#FF0000"), "zh-CN")
        self.assertEqual(name, "红色")

    def test_gibberish_returns_none(self):
        self.assertIsNone(color_from_description("qzxwvqq", "zh-CN"))


if __name__ == "__main__":
    unittest.main()
