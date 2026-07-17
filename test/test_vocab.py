import unittest

from ovos_color_parser import lookup_name, sRGBAColor
from ovos_color_parser.vocab import (palette_names, load_shared_palettes,
                                     load_locale_palettes, load_palettes)


class TestPaletteLoading(unittest.TestCase):
    def test_shared_webcolors_is_loaded(self):
        # webcolors.json lives at the res root and used to never load at all
        shared = load_shared_palettes()
        self.assertIn("webcolors", shared)
        self.assertTrue(shared["webcolors"])

    def test_locale_palettes_named(self):
        names = palette_names("en")
        self.assertIn("colors", names)
        self.assertIn("webcolors", names)  # shared merged in

    def test_locale_palettes_exclude_descriptors(self):
        self.assertNotIn("color_descriptors", load_locale_palettes("en"))

    def test_priority_order_is_deterministic(self):
        # common palettes come before niche catalogs, every call
        order1 = palette_names("en")
        order2 = palette_names("en")
        self.assertEqual(order1, order2)
        self.assertLess(order1.index("colors"), order1.index("RAL_classic"))

    def test_unknown_locale_empty(self):
        self.assertEqual(load_locale_palettes("xx"), {})
        # shared palettes still available via load_palettes
        self.assertIn("webcolors", load_palettes("xx"))


class TestNamespaceLookup(unittest.TestCase):
    def test_default_lookup(self):
        self.assertEqual(lookup_name(sRGBAColor.from_hex_str("#FF0000"), "en").lower(),
                         "red")

    def test_namespace_restricts(self):
        red = sRGBAColor.from_hex_str("#FF0000")
        web = lookup_name(red, "en", namespace="webcolors", nearest=True)
        cray = lookup_name(red, "en", namespace="crayola", nearest=True)
        self.assertNotEqual(web, cray)

    def test_unknown_namespace_raises(self):
        with self.assertRaises(ValueError):
            lookup_name(sRGBAColor(1, 2, 3), "en", namespace="does-not-exist")

    def test_nearest_fallback(self):
        # an off-palette color still resolves with nearest=True
        name = lookup_name(sRGBAColor(200, 30, 30), "en",
                           namespace="webcolors", nearest=True)
        self.assertTrue(name)

    def test_exact_miss_raises_without_nearest(self):
        with self.assertRaises(ValueError):
            lookup_name(sRGBAColor(13, 37, 42), "en")

    def test_deterministic(self):
        red = sRGBAColor.from_hex_str("#FF0000")
        self.assertEqual({lookup_name(red, "en") for _ in range(5)}, {lookup_name(red, "en")})


if __name__ == "__main__":
    unittest.main()
