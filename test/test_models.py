import unittest

from ovos_color_parser.models import (sRGBAColor, HSVColor, HLSColor, HueRange, SpectralColor,
                                      sRGBAColorPalette, HSVColorPalette, HLSColorPalette,
                                      SpectralColorPalette, ColorTerm, EnglishColorTerms,
                                      ISCCNBSSpectralColorTerms, NewtonSpectralColorTerms)


class TestSRGBAColor(unittest.TestCase):
    def test_from_hex_str_6_digit(self):
        c = sRGBAColor.from_hex_str("#1DA2DF")
        self.assertEqual((c.r, c.g, c.b, c.a), (29, 162, 223, 255))

    def test_from_hex_str_no_hash(self):
        c = sRGBAColor.from_hex_str("FF0000")
        self.assertEqual((c.r, c.g, c.b), (255, 0, 0))

    def test_from_hex_str_3_digit(self):
        c = sRGBAColor.from_hex_str("#F0A")
        self.assertEqual((c.r, c.g, c.b), (255, 0, 170))

    def test_from_hex_str_invalid_length(self):
        with self.assertRaises(ValueError):
            sRGBAColor.from_hex_str("#12345")

    def test_from_hex_str_invalid_chars(self):
        with self.assertRaises(ValueError):
            sRGBAColor.from_hex_str("#GGGGGG")

    def test_hex_str_roundtrip(self):
        for hex_code in ["#FF0000", "#00FF00", "#0000FF", "#1DA2DF", "#ABCDEF"]:
            self.assertEqual(sRGBAColor.from_hex_str(hex_code).hex_str, hex_code)

    def test_channel_validation(self):
        with self.assertRaises(ValueError):
            sRGBAColor(256, 0, 0)
        with self.assertRaises(ValueError):
            sRGBAColor(0, -1, 0)
        with self.assertRaises(ValueError):
            sRGBAColor(0, 0, 300)
        with self.assertRaises(ValueError):
            sRGBAColor(0, 0, 0, a=999)

    def test_hash_no_string_concat_collision(self):
        a = sRGBAColor(1, 11, 1)
        b = sRGBAColor(11, 1, 1)
        self.assertNotEqual(hash(a), hash(b))

    def test_hash_equal_colors(self):
        self.assertEqual(hash(sRGBAColor(1, 2, 3)), hash(sRGBAColor(1, 2, 3)))

    def test_usable_as_dict_key(self):
        d = {sRGBAColor(1, 2, 3): "x"}
        self.assertEqual(d[sRGBAColor(1, 2, 3)], "x")

    def test_hls_roundtrip(self):
        for hex_code in ["#FF0000", "#1DA2DF", "#808080", "#FFFFFF", "#000000", "#EE204D"]:
            c = sRGBAColor.from_hex_str(hex_code)
            self.assertEqual(c.as_hls.as_rgb.hex_str, hex_code)

    def test_hsv_roundtrip(self):
        for hex_code in ["#FF0000", "#1DA2DF", "#808080", "#FFFFFF", "#000000", "#EE204D"]:
            c = sRGBAColor.from_hex_str(hex_code)
            self.assertEqual(c.as_hsv.as_rgb.hex_str, hex_code)

    def test_metadata_survives_conversion(self):
        c = sRGBAColor(255, 0, 0, name="red", description="a red color")
        self.assertEqual(c.as_hls.name, "red")
        self.assertEqual(c.as_hsv.description, "a red color")


class TestHSVColor(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            HSVColor(361)
        with self.assertRaises(ValueError):
            HSVColor(-1)
        with self.assertRaises(ValueError):
            HSVColor(0, s=1.5)
        with self.assertRaises(ValueError):
            HSVColor(0, v=-0.1)

    def test_pure_red(self):
        c = HSVColor(0, 1.0, 1.0)
        self.assertEqual(c.as_rgb.hex_str, "#FF0000")

    def test_from_hex_str(self):
        c = HSVColor.from_hex_str("#00FF00")
        self.assertEqual(c.h, 120)


class TestHLSColor(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            HLSColor(400)
        with self.assertRaises(ValueError):
            HLSColor(0, l=2.0)
        with self.assertRaises(ValueError):
            HLSColor(0, s=-0.5)

    def test_pure_blue(self):
        c = HLSColor(240, 0.5, 1.0)
        self.assertEqual(c.as_rgb.hex_str, "#0000FF")

    def test_from_hex_str(self):
        c = HLSColor.from_hex_str("#0000FF")
        self.assertEqual(c.h, 240)


class TestHueRange(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            HueRange(-5, 30)
        with self.assertRaises(ValueError):
            HueRange(0, 400)

    def test_hue_midpoint(self):
        self.assertEqual(HueRange(100, 200).hue, 150)

    def test_as_spectral_color(self):
        sc = HueRange(0, 5).as_spectral_color
        self.assertIsInstance(sc, SpectralColor)
        self.assertEqual(sc.name, "Red")


class TestSpectralColor(unittest.TestCase):
    def test_wavelen_midpoint(self):
        sc = SpectralColor(wavelen_nm_min=600, wavelen_nm_max=700)
        self.assertEqual(sc.wavelen, 650)

    def test_as_rgb_uses_hex_approximation(self):
        sc = SpectralColor(wavelen_nm_min=600, wavelen_nm_max=700, hex_approximation="#FF0000")
        self.assertEqual(sc.as_rgb.hex_str, "#FF0000")

    def test_from_hex_str(self):
        sc = SpectralColor.from_hex_str("#FF0000")
        self.assertIsInstance(sc, SpectralColor)

    def test_out_of_palette_wavelength(self):
        sc = SpectralColor(wavelen_nm_min=10, wavelen_nm_max=11)
        with self.assertRaises(ValueError):
            sc.as_rgb


class TestPalettes(unittest.TestCase):
    def test_rgb_palette_conversions(self):
        p = sRGBAColorPalette(colors=[sRGBAColor(255, 0, 0), sRGBAColor(0, 0, 255)])
        self.assertIsInstance(p.as_hls, HLSColorPalette)
        self.assertIsInstance(p.as_hsv, HSVColorPalette)
        self.assertEqual(len(p.as_hls.colors), 2)
        self.assertEqual(p.as_hls.as_rgb.colors[0].hex_str, "#FF0000")
        self.assertEqual(p.as_hsv.as_rgb.colors[1].hex_str, "#0000FF")

    def test_spectral_palette_conversions(self):
        self.assertEqual(len(ISCCNBSSpectralColorTerms.as_rgb.colors),
                         len(ISCCNBSSpectralColorTerms.colors))
        self.assertTrue(NewtonSpectralColorTerms.colors)


class TestColorTerm(unittest.TestCase):
    def test_hue_derived_from_hex(self):
        term = ColorTerm("red", hex_approximation="#FF0000")
        self.assertIsNotNone(term.hue)
        self.assertLessEqual(term.hue.min_hue_approximation, 15)

    def test_as_rgb_from_hex(self):
        term = ColorTerm("red", hex_approximation="#FF0000")
        self.assertEqual(term.as_rgb.hex_str, "#FF0000")

    def test_english_terms_have_hex(self):
        for term in EnglishColorTerms.terms:
            self.assertIsNotNone(term.hex_approximation,
                                 f"{term.name} is missing hex_approximation")
            self.assertTrue(term.hex_approximation.startswith("#"))
            self.assertIsNotNone(term.hue)

    def test_english_terms_names(self):
        names = {t.name for t in EnglishColorTerms.terms}
        for expected in ("red", "orange", "yellow", "green", "cyan",
                         "blue", "purple", "magenta", "pink"):
            self.assertIn(expected, names)


if __name__ == "__main__":
    unittest.main()
