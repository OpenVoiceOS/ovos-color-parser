"""Pure-Python multi-pattern substring matcher.

Color matching needs to find every known name that appears inside a spoken
description ("make the lamp *moss green*" -> "moss green", "green"). A full
Aho-Corasick automaton solves that in one pass over the text but requires a
compiled dependency.

Descriptions are short and color names are bounded in length, so a windowed
lookup is both simpler and, in practice, faster here: for each start position we
test the substrings up to the longest stored key against a hash set. Cost scales
with the *text* length and longest key, not with the size of the vocabulary
(which can be thousands of entries), so a big palette costs nothing extra per
query. The surface deliberately mirrors the subset of the old automaton API the
library used — ``add_word`` / ``make_automaton`` / ``iter`` / ``len`` — so it is a
drop-in replacement.
"""
from typing import Dict, Iterator, Tuple


class SubstringMatcher:
    def __init__(self) -> None:
        self._words: Dict[str, str] = {}
        self._max_len = 0

    def add_word(self, key: str, value: str) -> None:
        """Register ``key`` (an already-normalised name) mapping to ``value``
        (its hex code). A later key silently wins on collision, matching how the
        vocabularies are layered."""
        if not key:
            return
        self._words[key] = value
        self._max_len = max(self._max_len, len(key))

    def make_automaton(self) -> None:
        """Present for API compatibility; the windowed matcher needs no build
        step, so this is a no-op."""

    def __len__(self) -> int:
        return len(self._words)

    def iter(self, text: str) -> Iterator[Tuple[int, str]]:
        """Yield ``(end_index, value)`` for every stored key occurring in ``text``.

        ``end_index`` is the index of the match's last character, mirroring the
        old automaton's contract so callers that unpack ``(_, value)`` keep
        working.
        """
        n = len(text)
        for start in range(n):
            end = min(start + self._max_len, n)
            for stop in range(start + 1, end + 1):
                candidate = text[start:stop]
                value = self._words.get(candidate)
                if value is not None:
                    yield stop - 1, value
