import json
import os

from quebra_frases import span_indexed_word_tokenize

from ovos_color_parser.color_helpers import Color, ColorOutOfSpace


def describe_color(color, lang):
    """
    Args:
        color: (Color): format for speech (True) or display (False)
        lang (str, optional): an optional BCP-47 language code, if omitted
                              the default language will be used.

    Returns:
        str: localized color description
    """
    resource_file = f"{os.path.dirname(__file__)}/webcolors.json"
    with open(resource_file) as f:
        COLORS = json.load(f)

    if color.hex in COLORS:
        return COLORS.get(color.hex)

    # fallback - just return main color
    color = color.main_color
    if color.hex in COLORS:
        return COLORS.get(color.hex)

    raise NotImplementedError


def get_color(text, lang):
    """
        Given a color description, return a Color object

        Args:
            text (str): the string describing a color
            lang (str, optional): an optional BCP-47 language code, if omitted
                              the default language will be used.
        Returns:
            (list): list of tuples with detected color and span of the
                    color in parent utterance [(Color, (start_idx, end_idx))]
        """
    resource_file = f"{os.path.dirname(__file__)}/webcolors.json"
    with open(resource_file) as f:
        COLORS = {v.lower(): k for k, v in json.load(f).items()}

    text = text.lower().strip()
    if text in COLORS:
        return Color.from_hex(COLORS[text])

    spans = extract_color_spans(text, lang)
    if spans:
        return spans[0][0]
    return ColorOutOfSpace()


def extract_color_spans(text, lang=''):
    """
        This function tags colors in an utterance.
        Args:
            text (str): the string to extract colors from
            lang (str, optional): an optional BCP-47 language code, if omitted
                              the default language will be used.
        Returns:
            (list): list of tuples with detected color and span of the
                    color in parent utterance [(Color, (start_idx, end_idx))]
        """
    resource_file = f"{os.path.dirname(__file__)}/webcolors.json"
    with open(resource_file) as f:
        COLORS = {v.lower(): k for k, v in json.load(f).items()}

    color_spans = []
    text = text.lower()
    spans = span_indexed_word_tokenize(text)

    for idx, (start, end, word) in enumerate(spans):
        next_span = spans[idx + 1] if idx + 1 < len(spans) else ()
        next_next_span = spans[idx + 2] if idx + 2 < len(spans) else ()
        word2 = word3 = ""
        if next_next_span:
            word3 = f"{word} {next_span[-1]} {next_next_span[-1]}"
        if next_span:
            word2 = f"{word} {next_span[-1]}"

        if next_span and next_next_span and word3 in COLORS:
            spans[idx + 1] = spans[idx + 2] = (-1, -1, "")
            end = next_next_span[1]
            color = Color.from_hex(COLORS[word3])
            color_spans.append((color, (start, end)))
        elif next_span and word2 in COLORS:
            spans[idx + 1] = (-1, -1, "")
            end = next_span[1]
            color = Color.from_hex(COLORS[word2])
            color_spans.append((color, (start, end)))
        elif word in COLORS:
            color = Color.from_hex(COLORS[word])
            color_spans.append((color, (start, end)))

    return color_spans
