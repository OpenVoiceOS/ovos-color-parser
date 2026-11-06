"""Hex codes with an alpha channel.

CSS Color 4, section 5.2 "The RGB Hexadecimal Notations", reads 3, 4, 6 or
8 hexadecimal digits. In the 8 digit form the last pair is the alpha
channel, where 00 is fully transparent and ff fully opaque; the 4 digit form
expands like the 3 digit one. A name describes the colour, so lookup_name
gives the same name with or without the alpha channel."""
import unittest

from ovos_color_parser import lookup_name, sRGBAColor


class TestHexAlpha(unittest.TestCase):

    def test_8_digits(self):
        c = sRGBAColor.from_hex_str("#0000ffcc")
        # cc = 12 * 16 + 12 = 204
        self.assertEqual((c.r, c.g, c.b, c.a), (0, 0, 255, 204))

    def test_4_digits(self):
        c = sRGBAColor.from_hex_str("#f008")
        # 8 expands to 88 = 8 * 16 + 8 = 136
        self.assertEqual((c.r, c.g, c.b, c.a), (255, 0, 0, 136))

    def test_6_digits_are_opaque(self):
        self.assertEqual(sRGBAColor.from_hex_str("#ff0000").a, 255)

    def test_lookup_name_ignores_alpha(self):
        for lang in ("en", "pt-PT", "de"):
            with self.subTest(lang=lang):
                self.assertEqual(
                    lookup_name(sRGBAColor.from_hex_str("#ff000080"), lang),
                    lookup_name(sRGBAColor.from_hex_str("#ff0000"), lang))
        self.assertEqual(
            lookup_name(sRGBAColor.from_hex_str("#ff000080"), "en"), "Red")

    def test_invalid_length_message(self):
        with self.assertRaisesRegex(ValueError, r"^Invalid hex string 12345$"):
            sRGBAColor.from_hex_str("#12345")


if __name__ == "__main__":
    unittest.main()
