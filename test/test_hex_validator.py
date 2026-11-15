"""is_hex_code_valid accepts the digit set that from_hex_str reads.

CSS Color 4, section 5.2, writes a hex colour as "#" and hexadecimal digits,
0-9 or a-f in either case. ``int(x, 16)`` also takes spaces, a sign,
underscores and other Unicode digits, so it cannot be the check. The
validator keeps its 3 and 6 digit lengths."""
import unittest

from ovos_color_parser import is_hex_code_valid, sRGBAColor

NOT_HEX = ("+fffff", " fffff", "fffff\t", "#١٢٣٤٥٦",
           "-fffff", "##fff", "#ff_f00")


class TestHexValidatorDigits(unittest.TestCase):

    def test_refuses_what_is_not_hex(self):
        for code in NOT_HEX:
            with self.subTest(code=code):
                self.assertFalse(is_hex_code_valid(code))

    def test_agrees_with_from_hex_str(self):
        for code in NOT_HEX:
            with self.subTest(code=code):
                with self.assertRaisesRegex(ValueError, r"^Invalid hex string "):
                    sRGBAColor.from_hex_str(code)

    def test_still_accepts_3_and_6_digits(self):
        for code in ("#FFF", "fff", "#ff0000", "FF0000", "#AbCdEf", "000"):
            with self.subTest(code=code):
                self.assertTrue(is_hex_code_valid(code))

    def test_still_refuses_4_and_8_digits(self):
        for code in ("#f008", "#ff000080"):
            with self.subTest(code=code):
                self.assertFalse(is_hex_code_valid(code))


if __name__ == "__main__":
    unittest.main()
