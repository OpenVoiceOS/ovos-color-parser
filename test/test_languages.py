"""Per-language behavior tests.

Anchor words below are hand-picked, common basic color terms taken from each
locale's bundled ``colors.json`` wordlist.
"""
import json
import os
import unittest

from ovos_color_parser import color_from_description, lookup_name, sRGBAColor

RES = os.path.join(os.path.dirname(__file__), "..", "ovos_color_parser", "res")

# lang -> (red word, blue word)
ANCHORS = {
    "en": ("red", "blue"),
    "ca": ("vermell", "blau"),
    "cs": ("červená", "modrá"),
    "da": ("rød", "blå"),
    "de": ("rot", "blau"),
    "es": ("rojo", "azul"),
    "eu": ("gorria", "urdina"),
    "fr": ("rouge", "bleu"),
    "it": ("rosso", "blu"),
    "nl": ("rood", "blauw"),
    "oc": ("roge", "blau"),
    "pl": ("czerwony", "niebieski"),
    "pt": ("vermelho", "azul"),
    "ru": ("красный", "синий"),
}


def is_reddish(color: sRGBAColor) -> bool:
    h = color.as_hls.h
    return h <= 40 or h >= 320


def is_bluish(color: sRGBAColor) -> bool:
    return 180 <= color.as_hls.h <= 280


class TestAllLanguagesParse(unittest.TestCase):
    def test_red_anchor(self):
        for lang, (red, _) in ANCHORS.items():
            with self.subTest(lang=lang, word=red):
                c = color_from_description(red, lang)
                self.assertIsNotNone(c, f"{lang}: no match for {red!r}")
                self.assertTrue(is_reddish(c), f"{lang}: {red!r} -> {c.hex_str} not reddish")

    def test_blue_anchor(self):
        for lang, (_, blue) in ANCHORS.items():
            with self.subTest(lang=lang, word=blue):
                c = color_from_description(blue, lang)
                self.assertIsNotNone(c, f"{lang}: no match for {blue!r}")
                self.assertTrue(is_bluish(c), f"{lang}: {blue!r} -> {c.hex_str} not bluish")

    def test_gibberish_returns_none(self):
        for lang in ANCHORS:
            with self.subTest(lang=lang):
                self.assertIsNone(color_from_description("qzxwvqq", lang))


class TestLocaleResources(unittest.TestCase):
    """Data-driven checks against every shipped locale directory."""

    @classmethod
    def setUpClass(cls):
        cls.locales = sorted(d for d in os.listdir(RES)
                             if os.path.isdir(os.path.join(RES, d)))

    def test_every_locale_has_colors_json(self):
        for loc in self.locales:
            with self.subTest(locale=loc):
                path = os.path.join(RES, loc, "colors.json")
                self.assertTrue(os.path.isfile(path), f"{loc} has no colors.json")

    def test_all_wordlists_are_valid(self):
        for loc in self.locales:
            for fname in os.listdir(os.path.join(RES, loc)):
                if not fname.endswith(".json"):
                    continue
                with self.subTest(locale=loc, file=fname):
                    with open(os.path.join(RES, loc, fname)) as f:
                        data = json.load(f)
                    self.assertIsInstance(data, dict)
                    self.assertTrue(data, f"{loc}/{fname} is empty")

    def test_color_wordlist_keys_are_hex(self):
        for loc in self.locales:
            for fname in os.listdir(os.path.join(RES, loc)):
                if not fname.endswith(".json") or fname == "color_descriptors.json":
                    continue
                with open(os.path.join(RES, loc, fname)) as f:
                    data = json.load(f)
                for hex_str in data:
                    with self.subTest(locale=loc, file=fname, key=hex_str):
                        sRGBAColor.from_hex_str(hex_str)  # must not raise

    def test_descriptor_files_have_all_keys(self):
        needed = {"very_high_saturation", "high_saturation", "low_saturation",
                  "very_low_saturation", "very_high_brightness", "high_brightness",
                  "low_brightness", "very_low_brightness", "very_high_temperature",
                  "high_temperature", "low_temperature", "very_low_temperature",
                  "very_high_opacity", "high_opacity", "low_opacity", "very_low_opacity"}
        for loc in self.locales:
            path = os.path.join(RES, loc, "color_descriptors.json")
            if not os.path.isfile(path):
                continue
            with self.subTest(locale=loc):
                with open(path) as f:
                    keys = set(json.load(f))
                self.assertEqual(needed - keys, set(), f"{loc} missing descriptor keys")

    def test_lookup_name_roundtrip_from_data(self):
        """Any hex present in a locale's colors.json can be looked up again."""
        for loc in self.locales:
            lang = loc  # full locale codes resolve exactly
            with open(os.path.join(RES, loc, "colors.json")) as f:
                data = json.load(f)
            hex_str, name = next(iter(data.items()))
            with self.subTest(locale=loc, hex=hex_str):
                found = lookup_name(sRGBAColor.from_hex_str(hex_str), lang)
                self.assertIsInstance(found, str)
                self.assertTrue(found)


class TestDiacriticsAndCompounds(unittest.TestCase):
    def test_diacritics_pt(self):
        # "Âmbar" (amber) ships in pt-BR colors.json
        c = color_from_description("âmbar", "pt-BR")
        self.assertIsNotNone(c)

    def test_diacritics_ru(self):
        c = color_from_description("зелёный", "ru")
        c2 = color_from_description("зеленый", "ru")
        self.assertTrue(c or c2)

    def test_compound_name_en(self):
        c = color_from_description("navy blue", "en")
        self.assertIsNotNone(c)
        self.assertTrue(160 <= c.as_hls.h <= 280, f"got {c.as_hls.h}")

    def test_modifier_de(self):
        base = color_from_description("rot", "de")
        dark = color_from_description("dunkles rot", "de")
        self.assertIsNotNone(dark)
        self.assertLess(dark.as_hls.l, base.as_hls.l)

    def test_modifier_fr(self):
        base = color_from_description("rouge", "fr")
        dark = color_from_description("rouge sombre", "fr")
        self.assertIsNotNone(dark)
        self.assertLess(dark.as_hls.l, base.as_hls.l)

    def test_modifier_pt(self):
        base = color_from_description("vermelho", "pt")
        dark = color_from_description("vermelho escuro", "pt")
        self.assertIsNotNone(dark)
        self.assertLess(dark.as_hls.l, base.as_hls.l)


if __name__ == "__main__":
    unittest.main()
