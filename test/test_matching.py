import unittest

from ovos_color_parser import (color_from_description, palette_from_description, lookup_name,
                               get_contrasting_black_or_white, color_distance, closest_color,
                               convert_K_to_RGB, average_colors, is_hex_code_valid,
                               sRGBAColor, HLSColor, sRGBAColorPalette)
from ovos_color_parser.matching import (rgb_to_cmyk, cmyk_to_rgb, ColorMatcher,
                                        _resolve_lang_dir, _norm)


class TestColorDistance(unittest.TestCase):
    def test_identical_colors(self):
        c = sRGBAColor(10, 20, 30)
        self.assertAlmostEqual(color_distance(c, sRGBAColor(10, 20, 30)), 0.0)

    def test_different_colors(self):
        self.assertGreater(color_distance(sRGBAColor(255, 0, 0), sRGBAColor(0, 0, 255)), 10)

    def test_accepts_hls(self):
        d = color_distance(HLSColor(0, 0.5, 1.0), sRGBAColor(255, 0, 0))
        self.assertAlmostEqual(d, 0.0, places=1)

    def test_closest_color(self):
        opts = [sRGBAColor(255, 0, 0), sRGBAColor(0, 255, 0), sRGBAColor(0, 0, 255)]
        best = closest_color(sRGBAColor(200, 30, 30), opts)
        self.assertEqual(best.hex_str, "#FF0000")


class TestLookupName(unittest.TestCase):
    def test_known_color(self):
        self.assertEqual(lookup_name(sRGBAColor.from_hex_str("#FF0000"), "en").lower(), "red")

    def test_unknown_color_raises(self):
        with self.assertRaises(ValueError):
            lookup_name(sRGBAColor(13, 37, 42), "en")

    def test_accepts_hls(self):
        self.assertEqual(lookup_name(HLSColor.from_hex_str("#FF0000"), "en").lower(), "red")


class TestColorFromDescription(unittest.TestCase):
    def test_simple_color(self):
        c = color_from_description("red", "en")
        self.assertIsInstance(c, sRGBAColor)
        hls = c.as_hls
        self.assertTrue(hls.h <= 30 or hls.h >= 330, f"expected reddish hue, got {hls.h}")

    def test_no_match_returns_none(self):
        self.assertIsNone(color_from_description("qzxwv", "en"))

    def test_unknown_language_returns_none(self):
        self.assertIsNone(color_from_description("red", "xx"))

    def test_dark_modifier_darkens(self):
        base = color_from_description("red", "en")
        dark = color_from_description("dark red", "en")
        self.assertLess(dark.as_hls.l, base.as_hls.l)

    def test_light_modifier_lightens(self):
        base = color_from_description("blue", "en")
        light = color_from_description("light blue", "en")
        self.assertGreater(light.as_hls.l, base.as_hls.l)

    def test_opacity_modifier_does_not_crash(self):
        c = color_from_description("transparent red", "en")
        self.assertIsInstance(c, sRGBAColor)
        self.assertLess(c.a, 255)

    def test_temperature_modifier_keeps_valid_channels(self):
        c = color_from_description("warm green", "en")
        self.assertIsInstance(c, sRGBAColor)
        self.assertTrue(0 <= c.r <= 255 and 0 <= c.g <= 255 and 0 <= c.b <= 255)

    def test_description_and_name_set(self):
        c = color_from_description("dark red", "en")
        self.assertEqual(c.description, "dark red")
        self.assertEqual(c.name, "Dark Red")

    def test_cast_to_palette_returns_candidate(self):
        c = color_from_description("dark red", "en", cast_to_palette=True)
        self.assertIsInstance(c, sRGBAColor)

    def test_object_color(self):
        # "carrot" is present in en-US object_colors.json
        c = color_from_description("carrot", "en")
        self.assertIsInstance(c, sRGBAColor)

    def test_case_insensitive(self):
        self.assertIsNotNone(color_from_description("RED", "en"))

    def test_full_locale_code(self):
        self.assertIsNotNone(color_from_description("red", "en-US"))
        self.assertIsNotNone(color_from_description("red", "en-GB"))


class TestPaletteFromDescription(unittest.TestCase):
    def test_returns_palette(self):
        p = palette_from_description("red", "en")
        self.assertIsInstance(p, sRGBAColorPalette)
        self.assertTrue(p.colors)

    def test_no_match_empty_palette(self):
        p = palette_from_description("qzxwv", "en")
        self.assertEqual(p.colors, [])


class TestContrast(unittest.TestCase):
    def test_light_background_gets_black(self):
        c = get_contrasting_black_or_white("#FFFFFF")
        self.assertEqual(c.hex_str, "#000000")
        self.assertEqual(c.name, "black")

    def test_dark_background_gets_white(self):
        c = get_contrasting_black_or_white("#000000")
        self.assertEqual(c.hex_str, "#FFFFFF")
        self.assertEqual(c.name, "white")

    def test_mid_tones(self):
        self.assertEqual(get_contrasting_black_or_white("#FFFF00").hex_str, "#000000")
        self.assertEqual(get_contrasting_black_or_white("#00008B").hex_str, "#FFFFFF")


class TestKelvin(unittest.TestCase):
    def test_warm_temperature(self):
        c = convert_K_to_RGB(2700)
        self.assertEqual(c.r, 255)
        self.assertGreater(c.g, c.b)

    def test_cool_temperature(self):
        c = convert_K_to_RGB(10000)
        self.assertEqual(c.b, 255)

    def test_out_of_range(self):
        with self.assertRaises(ValueError):
            convert_K_to_RGB(500)
        with self.assertRaises(ValueError):
            convert_K_to_RGB(50000)

    def test_bounds_are_valid(self):
        for k in (1000, 6600, 40000):
            c = convert_K_to_RGB(k)
            self.assertTrue(0 <= c.r <= 255 and 0 <= c.g <= 255 and 0 <= c.b <= 255)


class TestAverageColors(unittest.TestCase):
    def test_single_color(self):
        avg = average_colors([sRGBAColor(255, 0, 0)])
        self.assertEqual(avg.as_rgb.hex_str, "#FF0000")

    def test_two_colors(self):
        avg = average_colors([sRGBAColor(255, 0, 0), sRGBAColor(0, 0, 255)])
        self.assertIsInstance(avg, HLSColor)

    def test_circular_hue_mean(self):
        # 350 degrees and 10 degrees average to 0, not 180
        avg = average_colors([HLSColor(350, 0.5, 1.0), HLSColor(10, 0.5, 1.0)])
        self.assertTrue(avg.h <= 20 or avg.h >= 340, f"got hue {avg.h}")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            average_colors([])

    def test_mismatched_weights_raise(self):
        with self.assertRaises(ValueError):
            average_colors([sRGBAColor(255, 0, 0)], weights=[0.5, 0.5])


class TestHexValidation(unittest.TestCase):
    def test_valid(self):
        for code in ("#FF0000", "FF0000", "#abc", "abc", "#AbCdEf"):
            self.assertTrue(is_hex_code_valid(code), code)

    def test_invalid(self):
        for code in ("", "#12345", "#1234567", "xyz", "#GGG", "not a color"):
            self.assertFalse(is_hex_code_valid(code), code)


class TestCMYK(unittest.TestCase):
    def test_black(self):
        self.assertEqual(rgb_to_cmyk(0, 0, 0), (0, 0, 0, 100))

    def test_white(self):
        self.assertEqual(rgb_to_cmyk(255, 255, 255), (0.0, 0.0, 0.0, 0.0))

    def test_roundtrip(self):
        for rgb in [(255, 0, 0), (0, 255, 0), (0, 0, 255), (12, 200, 100)]:
            c, m, y, k = rgb_to_cmyk(*rgb)
            back = cmyk_to_rgb(c, m, y, k)
            for a, b in zip(rgb, back):
                self.assertAlmostEqual(a, b, delta=1)


class TestLangResolution(unittest.TestCase):
    def test_exact_locale(self):
        self.assertTrue(_resolve_lang_dir("en-US").endswith("en-US"))

    def test_case_insensitive(self):
        self.assertTrue(_resolve_lang_dir("en-us").endswith("en-US"))

    def test_primary_subtag(self):
        self.assertTrue(_resolve_lang_dir("en").endswith("en-US"))
        self.assertTrue(_resolve_lang_dir("de").endswith("de-DE"))

    def test_other_region_falls_back(self):
        self.assertTrue(_resolve_lang_dir("en-GB").endswith("en-US"))

    def test_underscore_format(self):
        self.assertTrue(_resolve_lang_dir("en_US").endswith("en-US"))

    def test_unknown_language(self):
        self.assertIsNone(_resolve_lang_dir("xx"))
        self.assertIsNone(_resolve_lang_dir(""))


class TestColorMatcher(unittest.TestCase):
    def test_match_returns_list(self):
        matches = ColorMatcher.match_color_automaton("red", "en")
        self.assertIsInstance(matches, list)
        # can be iterated more than once (a zip object could not)
        self.assertEqual(len(list(matches)), len(list(matches)))

    def test_object_match_unknown_lang_empty(self):
        self.assertEqual(list(ColorMatcher.match_object_automaton("red", "xx")), [])

    def test_norm(self):
        self.assertEqual(_norm("Dark-Red!"), "dark red")
        self.assertEqual(_norm("  Navy_Blue, "), "navy blue")


if __name__ == "__main__":
    unittest.main()
