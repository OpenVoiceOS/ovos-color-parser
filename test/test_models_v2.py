import unittest

from ovos_color_parser.models import (
    sRGBAColor, HSVColor, HLSColor, HueRange, SpectralColor,
    sRGBAColorPalette, HSVColorPalette, HLSColorPalette, SpectralColorPalette,
    ColorTerm, LanguageColorVocabulary,
    EnglishColorTerms, OtjihereroColorTerms,
    NewtonSpectralColorTerms, ISCCNBSSpectralColorTerms, MalacaraSpectralColorTerms,
    CRCHandbookSpectralColorTerms, ElectroMagneticSpectrum,
    IRSpectralColors, UVSpectralColors, VISIBLE_MIN_NM, VISIBLE_MAX_NM,
)


class TestConversionInvariants(unittest.TestCase):
    HEXES = ["#000000", "#FFFFFF", "#FF0000", "#00FF00", "#0000FF",
             "#808080", "#1DA2DF", "#EE204D", "#123456", "#ABCDEF"]

    def test_rgb_hls_rgb_roundtrip(self):
        for h in self.HEXES:
            self.assertEqual(sRGBAColor.from_hex_str(h).as_hls.as_rgb.hex_str, h)

    def test_rgb_hsv_rgb_roundtrip(self):
        for h in self.HEXES:
            self.assertEqual(sRGBAColor.from_hex_str(h).as_hsv.as_rgb.hex_str, h)

    def test_metadata_survives_all_conversions(self):
        c = sRGBAColor(10, 20, 30, name="n", description="d")
        for view in (c.as_hls, c.as_hsv, c.as_hls.as_hsv, c.as_hsv.as_hls):
            self.assertEqual(view.name, "n")
            self.assertEqual(view.description, "d")

    def test_hsv_hls_cross_roundtrip(self):
        for h in self.HEXES:
            c = sRGBAColor.from_hex_str(h)
            self.assertEqual(c.as_hsv.as_hls.as_rgb.hex_str, h)

    def test_alpha_preserved_through_hls(self):
        self.assertEqual(sRGBAColor(255, 0, 0, 128).a, 128)


class TestChannelValidationBoundaries(unittest.TestCase):
    def test_rgb_boundaries_ok(self):
        sRGBAColor(0, 0, 0)
        sRGBAColor(255, 255, 255, 0)
        sRGBAColor(255, 255, 255, 255)

    def test_hsv_boundaries_ok(self):
        HSVColor(0, 0.0, 0.0)
        HSVColor(360, 1.0, 1.0)

    def test_hls_boundaries_ok(self):
        HLSColor(0, 0.0, 0.0)
        HLSColor(360, 1.0, 1.0)

    def test_hue_range_boundaries_ok(self):
        HueRange(0, 360)
        HueRange(180, 180)


class TestSpectralVisibility(unittest.TestCase):
    def test_visible_bands(self):
        for c in ISCCNBSSpectralColorTerms.colors:
            self.assertTrue(c.is_visible, f"{c.name} should be visible")

    def test_infrared_not_visible(self):
        for c in IRSpectralColors.colors:
            self.assertFalse(c.is_visible, f"{c.name} should be non-visible")

    def test_ultraviolet_not_visible(self):
        for c in UVSpectralColors.colors:
            self.assertFalse(c.is_visible, f"{c.name} should be non-visible")

    def test_visibility_agrees_with_bounds(self):
        self.assertTrue(SpectralColor(VISIBLE_MIN_NM, VISIBLE_MIN_NM).is_visible)
        self.assertTrue(SpectralColor(VISIBLE_MAX_NM, VISIBLE_MAX_NM).is_visible)
        self.assertFalse(SpectralColor(VISIBLE_MAX_NM + 100, VISIBLE_MAX_NM + 200).is_visible)

    def test_as_rgb_still_returns_color_for_nonvisible(self):
        # placeholder RGB is honest-but-present; is_visible is the real signal
        c = IRSpectralColors.colors[0]
        self.assertEqual(c.as_rgb.hex_str, "#000000")
        self.assertFalse(c.is_visible)


class TestSpectralConversions(unittest.TestCase):
    def test_all_named_palettes_convert_to_rgb(self):
        for palette in (NewtonSpectralColorTerms, ISCCNBSSpectralColorTerms,
                        MalacaraSpectralColorTerms, CRCHandbookSpectralColorTerms):
            rgb = palette.as_rgb
            self.assertEqual(len(rgb.colors), len(palette.colors))
            for c in rgb.colors:
                self.assertTrue(0 <= c.r <= 255)

    def test_electromagnetic_spectrum_composed(self):
        self.assertEqual(len(ElectroMagneticSpectrum.colors),
                         len(IRSpectralColors.colors) + len(ISCCNBSSpectralColorTerms.colors)
                         + len(UVSpectralColors.colors))

    def test_hue_to_wavelength_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            SpectralColor(wavelen_nm_min=5, wavelen_nm_max=6).as_rgb

    def test_spectral_from_factories(self):
        self.assertIsInstance(SpectralColor.from_rgb(255, 0, 0), SpectralColor)
        self.assertIsInstance(SpectralColor.from_hsv(0, 1, 1), SpectralColor)
        self.assertIsInstance(SpectralColor.from_hls(0, 0.5, 1), SpectralColor)


class TestHueRangeSampling(unittest.TestCase):
    def test_sample_default_non_empty(self):
        self.assertTrue(HueRange(0, 120).sample().colors)

    def test_sample_hues_span_range(self):
        cols = HueRange(0, 120).sample(steps=3).colors
        self.assertEqual([c.h for c in cols], [0, 60, 120])

    def test_sample_single_step(self):
        self.assertEqual(len(HueRange(0, 120).sample(steps=1).colors), 1)

    def test_sample_zero_steps_clamped(self):
        self.assertEqual(len(HueRange(0, 120).sample(steps=0).colors), 1)

    def test_as_rgb_matches_sample_length(self):
        self.assertEqual(len(HueRange(0, 120).as_rgb.colors),
                         len(HueRange(0, 120).as_hsv.colors))

    def test_as_spectral_named(self):
        sc = HueRange(0, 5).as_spectral_color
        self.assertEqual(sc.name, "Red")


class TestPaletteConversions(unittest.TestCase):
    def test_all_palette_directions(self):
        p = sRGBAColorPalette(colors=[sRGBAColor(255, 0, 0), sRGBAColor(0, 255, 0)])
        self.assertIsInstance(p.as_hls, HLSColorPalette)
        self.assertIsInstance(p.as_hsv, HSVColorPalette)
        self.assertEqual(p.as_hls.as_rgb.colors[0].hex_str, "#FF0000")
        self.assertEqual(p.as_hsv.as_rgb.colors[1].hex_str, "#00FF00")

    def test_hsv_palette_roundtrip(self):
        p = HSVColorPalette(colors=[HSVColor(0, 1, 1), HSVColor(120, 1, 1)])
        self.assertEqual(p.as_rgb.colors[0].hex_str, "#FF0000")
        self.assertIsInstance(p.as_hls, HLSColorPalette)

    def test_hls_palette_roundtrip(self):
        p = HLSColorPalette(colors=[HLSColor(240, 0.5, 1.0)])
        self.assertEqual(p.as_rgb.colors[0].hex_str, "#0000FF")
        self.assertIsInstance(p.as_hsv, HSVColorPalette)


class TestColorTermVocab(unittest.TestCase):
    def test_hex_only_term_derives_hue(self):
        t = ColorTerm("x", hex_approximation="#00FF00")
        self.assertIsNotNone(t.hue)
        self.assertEqual(t.as_rgb.hex_str, "#00FF00")

    def test_hue_only_term_derives_hex(self):
        # a hue within spectral coverage (red band) derives a hex approximation
        t = ColorTerm("y", hue=HueRange(0, 5))
        self.assertIsNotNone(t.hex_approximation)
        self.assertTrue(t.hex_approximation.startswith("#"))

    def test_hue_only_term_uncovered_hue_leaves_hex_none(self):
        # hue outside the discontinuous spectral term coverage cannot derive a hex
        t = ColorTerm("z", hue=HueRange(30, 30))
        self.assertIsNone(t.hex_approximation)

    def test_english_terms_complete(self):
        for t in EnglishColorTerms.terms:
            self.assertTrue(t.hex_approximation.startswith("#"))
            self.assertIsNotNone(t.hue)

    def test_otjiherero_terms_have_hex(self):
        self.assertTrue(OtjihereroColorTerms.terms)
        for t in OtjihereroColorTerms.terms:
            self.assertEqual(t.as_rgb.hex_str[0], "#")

    def test_vocabulary_is_iterable(self):
        vocab = LanguageColorVocabulary(terms=[ColorTerm("a", hex_approximation="#000000")])
        self.assertEqual(len(vocab.terms), 1)


class TestHashingAndEquality(unittest.TestCase):
    def test_hash_ignores_metadata(self):
        self.assertEqual(hash(sRGBAColor(1, 2, 3, name="a")),
                         hash(sRGBAColor(1, 2, 3, name="b")))

    def test_alpha_affects_hash(self):
        self.assertNotEqual(hash(sRGBAColor(1, 2, 3, 255)),
                            hash(sRGBAColor(1, 2, 3, 128)))

    def test_dict_key_roundtrip(self):
        d = {sRGBAColor(9, 8, 7): "v"}
        self.assertEqual(d[sRGBAColor(9, 8, 7)], "v")


if __name__ == "__main__":
    unittest.main()
