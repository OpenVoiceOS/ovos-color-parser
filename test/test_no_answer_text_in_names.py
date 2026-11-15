"""A colour name is a name, never a sentence about a hex code.

Some translated palettes were built by asking a language model for the name of
each hex code. Where the model had no name, it answered with a sentence, such
as "Der er ingen specifik dansk farve for denne hex-kode." ("There is no
specific Danish colour for this hex code."). Such a sentence then matched as a
colour name. No name in any palette mentions a hex code or contains one.
"""
import glob
import json
import os
import re
import unittest

import ovos_color_parser

RES = os.path.join(os.path.dirname(ovos_color_parser.__file__), "res")
ANSWER = re.compile(r"(?i)\bhex|#[0-9a-f]{6}")


class TestNoAnswerTextInNames(unittest.TestCase):
    def test_no_name_mentions_a_hex_code(self):
        files = sorted(glob.glob(os.path.join(RES, "*", "*.json")))
        self.assertGreater(len(files), 0)
        checked = 0
        for path in files:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            for hex_code, name in data.items():
                if not isinstance(name, str):
                    continue
                checked += 1
                with self.subTest(file=os.path.relpath(path, RES), hex=hex_code):
                    self.assertIsNone(ANSWER.search(name), name)
        self.assertGreater(checked, 10000)

    def test_danish_hedged_answer_is_gone(self):
        path = os.path.join(RES, "da-DK", "colors.json")
        with open(path, encoding="utf-8") as f:
            names = json.load(f).values()
        self.assertFalse([n for n in names if n.startswith("Afhængig af belysningen")])


if __name__ == "__main__":
    unittest.main()
