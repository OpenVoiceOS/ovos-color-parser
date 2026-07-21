"""Per-language behavior tests.

Anchor words below are hand-picked, common basic color terms taken from each
locale's bundled ``colors.json`` wordlist.
"""
import json
import os
import unittest

from ovos_color_parser import color_from_description, lookup_name, sRGBAColor
from ovos_color_parser.matching import color_distance

RES = os.path.join(os.path.dirname(__file__), "..", "ovos_color_parser", "res")

# lang -> (red word, blue word)
ANCHORS = {
    "an": ("royo", "azul"),
    "ast": ("bermeyu", "azul"),
    "ar": ("أحمر", "أزرق"),
    "en": ("red", "blue"),
    "ca": ("vermell", "blau"),
    "cs": ("červená", "modrá"),
    "da": ("rød", "blå"),
    "de": ("rot", "blau"),
    "es": ("rojo", "azul"),
    "eu": ("gorria", "urdina"),
    "fr": ("rouge", "bleu"),
    "fy": ("read", "blau"),
    "it": ("rosso", "blu"),
    "kab": ("azeggaɣ", "anili"),
    "nl": ("rood", "blauw"),
    "oc": ("roge", "blau"),
    "pl": ("czerwony", "niebieski"),
    "pt": ("vermelho", "azul"),
    "ro": ("roșu", "albastru"),
    "ru": ("красный", "синий"),
    "sk": ("červená", "modrá"),
    "hr": ("crvena", "plava"),
    "bg": ("червено", "синьо"),
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


def is_greenish(color: sRGBAColor) -> bool:
    return 80 <= color.as_hls.h <= 170


class TestArabic(unittest.TestCase):
    """Comprehensive Modern Standard Arabic and dialectal coverage."""

    # Modern Standard Arabic named colors that must resolve with a plausible hue.
    MSA = {
        "أخضر": is_greenish,      # green
        "أصفر": None,             # yellow
        "برتقالي": None,          # orange
        "بنفسجي": None,           # purple
        "بني": None,              # brown
        "فيروزي": None,           # turquoise
        "أزرق مخضر": is_bluish,   # teal
        "سيان": is_bluish,        # cyan
        "مرجاني": is_reddish,     # coral
        "خزامي": None,            # lavender
        "زهري": is_reddish,       # pink
        "كاكي": None,             # khaki
        "خردلي": None,            # mustard
        "نعناعي": is_greenish,    # mint
        "كهرماني": None,          # amber
        "عنابي": is_reddish,      # burgundy
    }

    # Dialectal color terms grouped by major dialect area.
    DIALECT = {
        "بمبي": is_reddish,     # Egyptian pink
        "لبني": is_bluish,      # Egyptian/Levantine light blue
        "بترولي": is_bluish,    # Egyptian/Levantine petrol/teal
        "روز": is_reddish,      # Levantine/Maghrebi rose-pink
        "جوزي": None,           # Levantine/Iraqi walnut brown
        "قهوائي": is_reddish,   # Gulf coffee-brown
        "عسلي": None,           # Gulf honey
        "طوبي": is_reddish,     # Iraqi/Levantine brick-red
        "جكليتي": None,         # Iraqi/Gulf chocolate-brown
        "بلو": is_bluish,       # Maghrebi/Darija blue (French loan)
    }

    def _assert_hue(self, word, predicate):
        c = color_from_description(word, "ar")
        self.assertIsNotNone(c, f"no match for {word!r}")
        if predicate is not None:
            self.assertTrue(predicate(c), f"{word!r} -> {c.hex_str} hue {c.as_hls.h} unexpected")

    def test_msa_colors(self):
        for word, predicate in self.MSA.items():
            with self.subTest(word=word):
                self._assert_hue(word, predicate)

    def test_dialectal_colors(self):
        for word, predicate in self.DIALECT.items():
            with self.subTest(word=word):
                self._assert_hue(word, predicate)

    def test_orthographic_variants_hamzaless(self):
        # users type initial hamza forms interchangeably: أحمر vs احمر.
        # both spellings must land on perceptually the same color.
        for hamza, plain in [("أحمر", "احمر"), ("أزرق", "ازرق"),
                             ("أخضر", "اخضر"), ("أصفر", "اصفر"),
                             ("أبيض", "ابيض"), ("أسود", "اسود")]:
            with self.subTest(word=plain):
                a = color_from_description(hamza, "ar")
                b = color_from_description(plain, "ar")
                self.assertIsNotNone(b, f"{plain!r} did not match")
                self.assertLess(color_distance(a, b), 15,
                                f"{plain!r} -> {b.hex_str} far from {hamza!r} -> {a.hex_str}")

    def test_orthographic_variant_yaa(self):
        # final ya written as alef-maqsura: بنفسجى for بنفسجي
        self._assert_hue("بنفسجى", None)
        self._assert_hue("وردى", is_reddish)

    def test_tashkeel_is_ignored(self):
        # tashkeel (vowel marks) are optional: a diacritized spelling must resolve
        # to exactly the same color as its bare form, for every color word.
        for bare, diac in [("أحمر", "أَحْمَر"), ("أزرق", "أَزْرَق"),
                           ("أخضر", "أَخْضَر"), ("أصفر", "أَصْفَر"),
                           ("بنفسجي", "بَنَفْسَجِيّ")]:
            with self.subTest(word=bare):
                b = color_from_description(bare, "ar")
                d = color_from_description(diac, "ar")
                self.assertIsNotNone(d, f"diacritized {diac!r} did not match")
                self.assertEqual(b.hex_str, d.hex_str,
                                 f"{diac!r} -> {d.hex_str} != {bare!r} -> {b.hex_str}")

    def test_tashkeel_on_dialectal_and_modifiers(self):
        # stripping applies everywhere: dialectal names and modifier phrases too
        self._assert_hue("بَمْبِي", is_reddish)                 # vowelled Egyptian pink
        base = color_from_description("أحمر", "ar")
        dark = color_from_description("أَحْمَر غَامِق", "ar")   # fully vowelled "dark red"
        self.assertIsNotNone(dark)
        self.assertLess(dark.as_hls.l, base.as_hls.l)

    def test_object_colors(self):
        # prototypical objects imply their color
        self._assert_hue("سماء", is_bluish)     # sky -> blue
        self._assert_hue("موز", None)           # banana -> yellow
        self._assert_hue("طماطم", is_reddish)   # tomato -> red
        self._assert_hue("بندورة", is_reddish)  # tomato (Levantine) -> red

    def test_dark_modifier_lowers_lightness(self):
        base = color_from_description("أحمر", "ar")
        dark = color_from_description("أحمر غامق", "ar")
        self.assertIsNotNone(dark)
        self.assertLess(dark.as_hls.l, base.as_hls.l)

    def test_light_modifier_raises_lightness(self):
        base = color_from_description("أزرق", "ar")
        light = color_from_description("أزرق فاتح", "ar")
        self.assertIsNotNone(light)
        self.assertGreater(light.as_hls.l, base.as_hls.l)

    def test_adversarial_empty(self):
        self.assertIsNone(color_from_description("", "ar"))

    def test_adversarial_non_arabic_noise(self):
        self.assertIsNone(color_from_description("qzxwvqq lorem ipsum", "ar"))

    def test_adversarial_unknown_word(self):
        self.assertIsNone(color_from_description("سيارة كبيرة", "ar"))  # "big car"

    def test_adversarial_embedded_substring(self):
        # "بنيان" (building) contains "بني" (brown) but must not be read as a color
        self.assertIsNone(color_from_description("بنيان", "ar"))

    def test_adversarial_short_word_no_fuzzy_collision(self):
        # short color/object words must not fuzzy-match longer unrelated words:
        # "قدم" (foot) and "مقدم" (presenter) both contain "دم" (blood)
        self.assertIsNone(color_from_description("قدم", "ar"))
        self.assertIsNone(color_from_description("مقدم", "ar"))
        # the real two-letter word still resolves exactly
        self.assertIsNotNone(color_from_description("دم", "ar"))


if __name__ == "__main__":
    unittest.main()
