import unittest

from ovos_color_parser.core import (srgb8_to_linear, linear_to_srgb8, blend_linear,
                                    GamutPolicy, in_gamut, fit_to_gamut, LinearRGB,
                                    srgb8_distance, delta_e_cie2000)
from ovos_color_parser.models import HueRange, HSVColorPalette, sRGBAColorPalette, HLSColorPalette
from ovos_color_parser import average_colors, sRGBAColor, HLSColor


class TestLinearRoundTrip(unittest.TestCase):
    def test_byte_roundtrip_is_exact(self):
        for rgb in [(0, 0, 0), (255, 255, 255), (255, 0, 0),
                    (12, 200, 100), (1, 2, 3), (128, 128, 128)]:
            back = linear_to_srgb8(srgb8_to_linear(*rgb))[:3]
            self.assertEqual(back, rgb)

    def test_alpha_roundtrip(self):
        self.assertEqual(linear_to_srgb8(srgb8_to_linear(10, 20, 30, 128))[3], 128)


class TestLinearBlend(unittest.TestCase):
    def test_black_white_midpoint_is_perceptual(self):
        # gamma-naive averaging gives 128; correct linear-light midpoint is ~188
        mix = linear_to_srgb8(blend_linear([srgb8_to_linear(0, 0, 0),
                                            srgb8_to_linear(255, 255, 255)]))
        self.assertEqual(mix[:3], (188, 188, 188))

    def test_weights_bias_result(self):
        heavy_red = linear_to_srgb8(blend_linear(
            [srgb8_to_linear(255, 0, 0), srgb8_to_linear(0, 0, 255)],
            weights=[3, 1]))
        self.assertGreater(heavy_red[0], heavy_red[2])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            blend_linear([])

    def test_mismatched_weights_raise(self):
        with self.assertRaises(ValueError):
            blend_linear([srgb8_to_linear(0, 0, 0)], weights=[1, 2])

    def test_zero_total_weight_is_mean(self):
        # weights summing to zero must not divide-by-zero
        out = linear_to_srgb8(blend_linear(
            [srgb8_to_linear(255, 0, 0), srgb8_to_linear(0, 0, 255)], weights=[0, 0]))
        self.assertTrue(all(0 <= c <= 255 for c in out))


class TestGamut(unittest.TestCase):
    def test_in_gamut(self):
        self.assertTrue(in_gamut(srgb8_to_linear(10, 20, 30)))
        self.assertFalse(in_gamut(LinearRGB(1.4, -0.1, 0.2)))

    def test_clamp_flags_and_fits(self):
        fitted, oog = fit_to_gamut(LinearRGB(1.4, -0.1, 0.2), GamutPolicy.CLAMP)
        self.assertTrue(oog)
        self.assertTrue(in_gamut(fitted))

    def test_map_preserves_gamut_and_flags(self):
        fitted, oog = fit_to_gamut(LinearRGB(1.4, -0.1, 0.2), GamutPolicy.MAP)
        self.assertTrue(oog)
        self.assertTrue(in_gamut(fitted))

    def test_reject_raises(self):
        with self.assertRaises(ValueError):
            fit_to_gamut(LinearRGB(1.4, -0.1, 0.2), GamutPolicy.REJECT)

    def test_in_gamut_input_not_flagged(self):
        _, oog = fit_to_gamut(srgb8_to_linear(10, 20, 30))
        self.assertFalse(oog)


class TestPerceptualDistance(unittest.TestCase):
    def test_ciede2000_reference_data(self):
        # Sharma, Wu & Dalal (2005) verification pairs
        cases = [
            ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
            ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
            ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
            ((22.7233, 20.0904, -46.694), (23.0331, 14.973, -42.5619), 2.0373),
        ]
        for lab1, lab2, expected in cases:
            self.assertAlmostEqual(delta_e_cie2000(lab1, lab2), expected, places=2)

    def test_identical_is_zero(self):
        self.assertEqual(srgb8_distance((10, 20, 30), (10, 20, 30)), 0.0)

    def test_ordering(self):
        # red is nearer to orange-ish than to blue
        self.assertLess(srgb8_distance((255, 0, 0), (200, 30, 30)),
                        srgb8_distance((255, 0, 0), (0, 0, 255)))


class TestAverageColorsLinear(unittest.TestCase):
    def test_single_color_preserved(self):
        self.assertEqual(average_colors([sRGBAColor(255, 0, 0)]).as_rgb.hex_str, "#FF0000")

    def test_description_has_no_container_repr(self):
        desc = average_colors([sRGBAColor(255, 0, 0, name="red"),
                               sRGBAColor(0, 0, 255, name="blue")]).description
        self.assertNotIn("{", desc)
        self.assertIn("red", desc)
        self.assertIn("blue", desc)

    def test_returns_hls(self):
        self.assertIsInstance(average_colors([sRGBAColor(255, 0, 0),
                                              sRGBAColor(0, 0, 255)]), HLSColor)


class TestHueRangePalette(unittest.TestCase):
    def test_as_hsv_is_non_empty(self):
        p = HueRange(0, 60).as_hsv
        self.assertIsInstance(p, HSVColorPalette)
        self.assertTrue(p.colors)

    def test_as_rgb_and_hls_non_empty(self):
        self.assertIsInstance(HueRange(0, 60).as_rgb, sRGBAColorPalette)
        self.assertIsInstance(HueRange(0, 60).as_hls, HLSColorPalette)
        self.assertTrue(HueRange(0, 60).as_rgb.colors)

    def test_sample_step_count(self):
        self.assertEqual(len(HueRange(0, 90).sample(steps=4).colors), 4)

    def test_zero_width_range_single_color(self):
        self.assertEqual(len(HueRange(120, 120).sample().colors), 1)


if __name__ == "__main__":
    unittest.main()
