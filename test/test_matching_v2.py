import unittest

from ovos_color_parser import (color_from_description, palette_from_description,
                               GamutPolicy, ColorMatcher, sRGBAColor, convert_K_to_RGB,
                               rgb_to_cmyk, cmyk_to_rgb, is_hex_code_valid)
from ovos_color_parser.core import in_gamut, srgb8_to_linear
from ovos_color_parser.vocab import palette_names


def _locales():
    # every locale that ships color vocabularies
    return [n for n in ("en", "en-US", "de-DE", "fr-FR", "es-ES", "it-IT", "nl-NL",
                        "pt-BR", "ru-RU", "pl-PL", "cs-CZ", "ca-ES", "eu-ES",
                        "da-DK", "ro-RO", "sk-SK", "hr-HR", "bg-BG", "oc-FR",
                        "an-ES", "ast-ES", "fy-NL", "kab-DZ", "ar-SA")
            if palette_names(n)]


class TestInstantiableMatcher(unittest.TestCase):
    def test_instance_matches_classmethod(self):
        inst = [c.hex_str for c, _ in ColorMatcher("en").match_colors("red")]
        cls = [c.hex_str for c, _ in ColorMatcher.match_color_automaton("red", "en")]
        self.assertEqual(sorted(inst), sorted(cls))

    def test_custom_vocab_injection(self):
        m = ColorMatcher("xx", color_palettes=[{"#FF0000": "rood", "#0000FF": "blauw"}])
        hits = {c.name for c, _ in m.match_colors("maak het rood")}
        self.assertEqual(hits, {"rood"})

    def test_custom_object_vocab(self):
        # matches are returned as (lossy) HLSColor, so assert on the resolved name
        m = ColorMatcher("xx", color_palettes=[{}], object_colors={"#F28C28": "wortel"})
        hits = {c.name for c, _ in m.match_objects("de wortel")}
        self.assertEqual(hits, {"wortel"})

    def test_empty_vocab_no_matches(self):
        m = ColorMatcher("xx", color_palettes=[{}])
        self.assertEqual(m.match_colors("anything"), [])

    def test_fuzzy_augments_exact(self):
        m = ColorMatcher("en")
        exact = m.match_colors("red", fuzzy=False)
        fuzzy = m.match_colors("red", fuzzy=True)
        self.assertGreaterEqual(len(fuzzy), len(exact))

    def test_no_duplicate_hex_name_pairs(self):
        m = ColorMatcher("en")
        pairs = [(c.hex_str, c.name) for c, _ in m.match_colors("red", fuzzy=True)]
        self.assertEqual(len(pairs), len(set(pairs)))


class TestGamutPolicyParam(unittest.TestCase):
    def test_result_always_in_gamut(self):
        for policy in (GamutPolicy.CLAMP, GamutPolicy.MAP):
            c = color_from_description("warm bright red", "en", gamut=policy)
            self.assertTrue(in_gamut(srgb8_to_linear(c.r, c.g, c.b)))

    def test_channels_valid_all_policies(self):
        for policy in (GamutPolicy.CLAMP, GamutPolicy.MAP):
            c = color_from_description("very warm green", "en", gamut=policy)
            self.assertTrue(0 <= c.r <= 255 and 0 <= c.g <= 255 and 0 <= c.b <= 255)


class TestAdversarialParsing(unittest.TestCase):
    def test_empty_string(self):
        self.assertIsNone(color_from_description("", "en"))

    def test_whitespace_only(self):
        self.assertIsNone(color_from_description("   ", "en"))

    def test_punctuation_only(self):
        self.assertIsNone(color_from_description("!?.,;", "en"))

    def test_very_long_description_does_not_crash(self):
        # long input must be handled without error (name-vs-whole-text scoring
        # may legitimately return None); the contract here is "no exception"
        try:
            color_from_description("please make the lamp a nice red " * 20, "en")
        except Exception as e:  # noqa: BLE001 - robustness guard
            self.fail(f"long input raised {e!r}")

    def test_short_realistic_command_parses(self):
        self.assertIsNotNone(color_from_description("make the lamp red", "en"))

    def test_numbers_only(self):
        self.assertIsNone(color_from_description("12345", "en"))

    def test_unicode_noise(self):
        self.assertIsNone(color_from_description("日本語のテキスト", "en"))

    def test_unknown_locale_returns_none(self):
        self.assertIsNone(color_from_description("red", "zz"))

    def test_palette_empty_on_no_match(self):
        self.assertEqual(palette_from_description("qzxwv", "en").colors, [])

    def test_leading_trailing_spaces(self):
        self.assertIsNotNone(color_from_description("   red   ", "en"))


class TestMultilingualSmoke(unittest.TestCase):
    def test_every_locale_loads_and_parses_something(self):
        # each shipped locale must resolve at least one of its own color names
        failures = []
        for lang in _locales():
            # pull a real name from the locale's own vocabulary
            from ovos_color_parser.vocab import load_locale_palettes
            palettes = load_locale_palettes(lang)
            sample_name = None
            for pal in palettes.values():
                if pal:
                    sample_name = next(iter(pal.values()))
                    break
            if sample_name is None:
                continue
            if color_from_description(sample_name, lang) is None:
                failures.append((lang, sample_name))
        self.assertEqual(failures, [], f"locales failing to parse own names: {failures}")

    def test_locales_have_palettes(self):
        self.assertGreaterEqual(len(_locales()), 10)


class TestKelvinMonotonic(unittest.TestCase):
    def test_warm_is_red_dominant(self):
        self.assertEqual(convert_K_to_RGB(2000).r, 255)

    def test_cool_is_blue_saturated(self):
        self.assertEqual(convert_K_to_RGB(20000).b, 255)

    def test_blue_increases_with_temperature(self):
        low = convert_K_to_RGB(3000)
        high = convert_K_to_RGB(9000)
        self.assertLessEqual(low.b, high.b)

    def test_out_of_range_raises(self):
        for k in (0, 999, 40001, 999999):
            with self.assertRaises(ValueError):
                convert_K_to_RGB(k)


class TestCMYKRoundTrip(unittest.TestCase):
    def test_roundtrip_many(self):
        for rgb in [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0),
                    (0, 0, 255), (12, 200, 100), (77, 88, 99)]:
            back = cmyk_to_rgb(*rgb_to_cmyk(*rgb))
            for a, b in zip(rgb, back):
                self.assertAlmostEqual(a, b, delta=1)


class TestHexValidationExhaustive(unittest.TestCase):
    def test_valid_forms(self):
        for code in ("#FFF", "FFF", "#FFFFFF", "FFFFFF", "#abc", "#AbCdEf", "000"):
            self.assertTrue(is_hex_code_valid(code), code)

    def test_invalid_forms(self):
        for code in ("", "#", "#FF", "#FFFF", "#FFFFF", "#FFFFFFF",
                     "#GGG", "xyz", "red", "#12 34 56"):
            self.assertFalse(is_hex_code_valid(code), code)


if __name__ == "__main__":
    unittest.main()
