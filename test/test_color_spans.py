import os
import re
import unittest

from ovos_color_parser import extract_color_spans, ColorSpan
from ovos_color_parser.matching import _color_name_maps

RES = os.path.join(os.path.dirname(__file__), "..", "ovos_color_parser", "res")


def _assert_invariant(case, text, spans):
    for span in spans:
        case.assertEqual(text[span.start:span.end], span.surface)
        case.assertTrue(re.match(r"^#[0-9a-f]{6}$", span.hex), span.hex)


class TestExtractColorSpansEnglish(unittest.TestCase):
    def test_one_color(self):
        text = "the car is red"
        spans = extract_color_spans(text, "en")
        _assert_invariant(self, text, spans)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].surface, "red")
        self.assertEqual(spans[0].hex, "#ff0000")

    def test_two_colors(self):
        text = "red and blue"
        spans = extract_color_spans(text, "en")
        _assert_invariant(self, text, spans)
        self.assertEqual([s.surface for s in spans], ["red", "blue"])
        self.assertEqual(spans, sorted(spans, key=lambda s: s.start))

    def test_modifier_phrase_is_one_span(self):
        text = "give me a light blue shirt"
        spans = extract_color_spans(text, "en")
        _assert_invariant(self, text, spans)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].surface, "light blue")
        self.assertEqual(spans[0].name, "Light Blue")
        plain = extract_color_spans("give me a blue shirt", "en")
        self.assertNotEqual(spans[0].hex, plain[0].hex)

    def test_color_word_inside_another_word_is_not_a_match(self):
        text = "please redirect the greenhouse"
        spans = extract_color_spans(text, "en")
        self.assertEqual(spans, [])

    def test_no_color(self):
        self.assertEqual(extract_color_spans("nothing to see here", "en"), [])

    def test_empty_string(self):
        self.assertEqual(extract_color_spans("", "en"), [])

    def test_color_at_start_and_end(self):
        text = "red at the start and at the end blue"
        spans = extract_color_spans(text, "en")
        _assert_invariant(self, text, spans)
        self.assertEqual(spans[0].surface, "red")
        self.assertEqual(spans[0].start, 0)
        self.assertEqual(spans[-1].surface, "blue")
        self.assertEqual(spans[-1].end, len(text))

    def test_leading_emoji_and_accented_word(self):
        text = "\U0001F3A8 café colored azul but also blue"
        spans = extract_color_spans(text, "en")
        _assert_invariant(self, text, spans)
        self.assertIn("blue", [s.surface for s in spans])

    def test_punctuation_adjacency(self):
        text = "blue, please"
        spans = extract_color_spans(text, "en")
        _assert_invariant(self, text, spans)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].surface, "blue")
        self.assertEqual(spans[0].end, 4)

    def test_hex_is_lowercase_and_name_is_canonical(self):
        spans = extract_color_spans("bright orange", "en")
        self.assertEqual(spans[0].hex, "#ff5b00")
        self.assertEqual(spans[0].name, "Bright Orange")

    def test_typo_never_produces_a_span(self):
        # fuzzy matching is out of scope for spans: a misspelled color name
        # is simply not found, unlike color_from_description(fuzzy=True)
        self.assertEqual(extract_color_spans("blu car", "en"), [])


class TestExtractColorSpansSpanish(unittest.TestCase):
    def test_one_color(self):
        text = "el coche es rojo"
        spans = extract_color_spans(text, "es")
        _assert_invariant(self, text, spans)
        self.assertEqual(spans[0].surface, "rojo")
        self.assertEqual(spans[0].hex, "#ff0000")

    def test_modifier_phrase_is_one_span(self):
        text = "quiero un azul claro"
        spans = extract_color_spans(text, "es")
        _assert_invariant(self, text, spans)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].surface, "azul claro")

    def test_color_word_inside_another_word_is_not_a_match(self):
        self.assertEqual(extract_color_spans("redireccionar el flujo", "es"), [])

    def test_no_color(self):
        self.assertEqual(extract_color_spans("no hay nada aqui", "es"), [])

    def test_punctuation_adjacency(self):
        text = "azul, por favor"
        spans = extract_color_spans(text, "es")
        _assert_invariant(self, text, spans)
        self.assertEqual(spans[0].surface, "azul")


class TestExtractColorSpansGerman(unittest.TestCase):
    def test_one_color(self):
        text = "das auto ist rot"
        spans = extract_color_spans(text, "de")
        _assert_invariant(self, text, spans)
        self.assertEqual(spans[0].surface, "rot")
        self.assertEqual(spans[0].hex, "#ff0000")

    def test_two_colors(self):
        text = "rot und blau"
        spans = extract_color_spans(text, "de")
        _assert_invariant(self, text, spans)
        self.assertEqual([s.surface for s in spans], ["rot", "blau"])

    def test_compound_color_word(self):
        text = "ich mag hellblau"
        spans = extract_color_spans(text, "de")
        _assert_invariant(self, text, spans)
        self.assertEqual(spans[0].surface, "hellblau")

    def test_color_word_inside_another_word_is_not_a_match(self):
        self.assertEqual(extract_color_spans("umleiten", "de"), [])

    def test_no_color(self):
        self.assertEqual(extract_color_spans("hier gibt es nichts", "de"), [])


class TestExtractColorSpansHyphenated(unittest.TestCase):
    def test_hyphenated_table_entry_is_one_span(self):
        text = "off-white walls"
        spans = extract_color_spans(text, "en")
        _assert_invariant(self, text, spans)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].surface, "off-white")
        self.assertEqual(spans[0].name, "Off White")

    def test_hyphenated_compound_color_names(self):
        text = "a blue-green sweater"
        spans = extract_color_spans(text, "en")
        _assert_invariant(self, text, spans)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].surface, "blue-green")
        self.assertEqual(spans[0].name, "Blue Green")

    def test_extra_whitespace_between_words_still_forms_a_phrase(self):
        text = "light  blue shirt"
        spans = extract_color_spans(text, "en")
        _assert_invariant(self, text, spans)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].surface, "light  blue")
        self.assertEqual(spans[0].name, "Light Blue")


class TestExtractColorSpansCJK(unittest.TestCase):
    def test_chinese_color_in_unspaced_sentence(self):
        text = "这是红色"
        spans = extract_color_spans(text, "zh-cn")
        _assert_invariant(self, text, spans)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].surface, "红色")

    def test_chinese_no_color(self):
        self.assertEqual(extract_color_spans("这是一本书", "zh-cn"), [])

    def test_japanese_color_in_unspaced_sentence(self):
        text = "赤い車"
        spans = extract_color_spans(text, "ja-jp")
        _assert_invariant(self, text, spans)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].surface, "赤")

    def test_single_ideograph_rejected_next_to_another_ideograph(self):
        # "赤字" (deficit): 赤 is immediately followed by another ideograph,
        # so the one-ideograph colour entry is not accepted here
        self.assertEqual(extract_color_spans("赤字", "ja-jp"), [])

    def test_single_ideograph_rejected_next_to_another_ideograph_2(self):
        # "黒板" (blackboard): 黒 followed by the ideograph 板
        self.assertEqual(extract_color_spans("黒板", "ja-jp"), [])

    def test_single_ideograph_next_to_kana_is_a_known_limitation(self):
        # "赤ちゃん" (baby): 赤 is followed by kana, not an ideograph, so it
        # matches the same way the genuine "赤い" ("red") does above -- the
        # two are not distinguishable without a real CJK segmenter, and the
        # matcher deliberately does not attempt one.
        text = "赤ちゃん"
        spans = extract_color_spans(text, "ja-jp")
        _assert_invariant(self, text, spans)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].surface, "赤")

    def test_multi_ideograph_entry_matches_anywhere(self):
        text = "红色的车"
        spans = extract_color_spans(text, "zh-cn")
        _assert_invariant(self, text, spans)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].surface, "红色")


class TestColorTableHexInvariant(unittest.TestCase):
    def test_every_language_table_normalizes_to_rrggbb(self):
        langs = [d for d in os.listdir(RES) if os.path.isdir(os.path.join(RES, d))]
        self.assertTrue(langs)
        for lang in langs:
            word_map, substring_map = _color_name_maps(lang)
            for hex_str, _name in list(word_map.values()) + list(substring_map.values()):
                self.assertRegex(hex_str, r"^#[0-9a-f]{6}$", f"{lang}: {hex_str}")

    def test_russian_three_digit_shorthand_is_expanded(self):
        text = "чёрный кот"
        spans = extract_color_spans(text, "ru-ru")
        _assert_invariant(self, text, spans)
        self.assertEqual(spans[0].surface, "чёрный")
        self.assertEqual(spans[0].hex, "#000000")


if __name__ == "__main__":
    unittest.main()
