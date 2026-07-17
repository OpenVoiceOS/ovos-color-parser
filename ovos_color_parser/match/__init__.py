"""Name-spotting machinery.

:class:`SubstringMatcher` is a small pure-Python stand-in for the multi-pattern
matcher the library used to get from ``pyahocorasick``. It finds every stored
color/object name that occurs anywhere in a description. Removing the compiled C
extension keeps the package pure-Python and installable without a build step.
"""
from ovos_color_parser.match.automaton import SubstringMatcher

__all__ = ["SubstringMatcher"]
