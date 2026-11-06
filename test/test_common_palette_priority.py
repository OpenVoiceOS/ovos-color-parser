import unittest

from ovos_color_parser import extract_color_spans

# Expected values are read by hand from the first palette in priority order
# that defines the name: res/<locale>/colors.json, or res/<locale>/99colors.json
# when colors.json lacks it. Within one file the first entry wins.
CASES = (
    # (locale, text, expected hex, source palette)
    ("de-DE", "grün", "#00ff00", "99colors"),
    ("de-DE", "rot", "#ff0000", "99colors"),
    ("en-US", "dark red", "#8b0000", "colors"),
    ("en-US", "light blue", "#95d0fc", "colors"),
    ("en-US", "sky blue", "#75bbfd", "colors"),
    ("en-US", "red", "#ff0000", "colors"),
    ("da-DK", "blå", "#0000ff", "colors"),
    ("sv-SE", "blå", "#0000ff", "colors"),
    ("nl-NL", "blauw", "#0000ff", "colors"),
)


class TestCommonPaletteWins(unittest.TestCase):
    def test_basic_names_use_the_common_palette(self):
        for lang, text, expected, source in CASES:
            with self.subTest(lang=lang, text=text, source=source):
                spans = extract_color_spans(text, lang)
                self.assertEqual(len(spans), 1)
                self.assertEqual(spans[0].hex.lower(), expected)


if __name__ == "__main__":
    unittest.main()
