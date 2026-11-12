"""A basic colour word casts to the colour that en-US gives its English term.

it-IT, da-DK, es-ES and eu-ES translate res/en-US/colors.json entry by entry,
and a cast returns the first entry whose name is the whole description. Each
word below is the locale's own translation of the en-US entry for the bare
English term (for example it-IT "#FFFFFF": "Bianco" and en-US "#FFFFFF": "White").
The expected hex is the first entry with that English name in en-US
colors.json, which is what the en-US cast of the English word returns. It is
read from the en-US file, not from the parser.

English has two spellings of grey, and en-US gives each one its own first
entry. The expected entry is the first of the two in en-US order that the
locale calls by its bare word: es-ES calls "#808080" "Cenizo" and it-IT calls
it "Grigio 50%", so both cast to the "Grey" entry.

Italian "marrone" is not the en-US "Brown". it.wikipedia "Marrone" gives
#804000 (128, 64, 0) in its infobox and lists #A52A2A only as "Rosso Marrone",
the web colour Brown. The hand-picked it-IT entry "Marrone" #964B00 comes first
and is the nearer of the two to #804000.
"""
import json
import os

import pytest

import ovos_color_parser
from ovos_color_parser import color_from_description

RES = os.path.join(os.path.dirname(ovos_color_parser.__file__), "res")


def en_us_hex(english):
    with open(os.path.join(RES, "en-US", "colors.json"), encoding="utf-8") as f:
        return next(h.upper() for h, n in json.load(f).items() if n == english)


def cast(word, lang):
    return color_from_description(word, lang, cast_to_palette=True).hex_str


@pytest.mark.parametrize("word, english", [
    ("bianco", "White"),
    ("nero", "Black"),
    ("rosso", "Red"),
    ("verde", "Green"),
    ("blu", "Blue"),
    ("grigio", "Grey"),
    ("rosa", "Pink"),
    ("viola", "Purple"),
])
def test_italian_basic_words(word, english):
    assert cast(word, "it-IT") == en_us_hex(english)


def rgb_distance(a, b):
    a, b = a.lstrip("#"), b.lstrip("#")
    return sum((int(a[i:i + 2], 16) - int(b[i:i + 2], 16)) ** 2 for i in (0, 2, 4)) ** 0.5


def test_italian_marrone_is_the_wikipedia_marrone():
    wikipedia_marrone = "#804000"
    got = cast("marrone", "it-IT")
    assert got == "#964B00"
    assert rgb_distance(got, wikipedia_marrone) < rgb_distance(en_us_hex("Brown"), wikipedia_marrone)


@pytest.mark.parametrize("word, english", [
    ("hvid", "White"),
    ("sort", "Black"),
    ("rød", "Red"),
    pytest.param("grøn", "Green", marks=pytest.mark.xfail(
        strict=True, reason='da-DK calls "#00FF00" "Limegrøn", not "Grøn"')),
    ("blå", "Blue"),
    ("brun", "Brown"),
    ("grå", "Gray"),
    ("lyserød", "Pink"),
    ("lilla", "Purple"),
    ("orange", "Orange"),
])
def test_danish_basic_words(word, english):
    assert cast(word, "da-DK") == en_us_hex(english)


@pytest.mark.parametrize("word, english", [
    ("blanco", "White"),
    pytest.param("negro", "Black", marks=pytest.mark.xfail(
        strict=True, reason='es-ES calls "#000000" "Sable", not "Negro"')),
    ("rojo", "Red"),
    ("verde", "Green"),
    ("azul", "Blue"),
    ("marrón", "Brown"),
    ("gris", "Grey"),
    ("rosa", "Pink"),
    ("amarillo", "Yellow"),
    ("naranja", "Orange"),
    ("púrpura", "Purple"),
])
def test_spanish_basic_words(word, english):
    assert cast(word, "es-ES") == en_us_hex(english)


@pytest.mark.parametrize("word, english", [
    ("laranja", "Orange"),
])
def test_basque_basic_words(word, english):
    assert cast(word, "eu-ES") == en_us_hex(english)
