"""Color namespaces, gamut handling and isolated matching."""
from ovos_color_parser import (lookup_name, palette_names, sRGBAColor,
                               color_from_description, GamutPolicy, ColorMatcher)

# each wordlist is a named namespace
print(palette_names("en")[:6])

# name a color in a chosen palette, or take the perceptually nearest match
red = sRGBAColor.from_hex_str("#FF0000")
print(lookup_name(red, "en"))                                    # Red
print(lookup_name(red, "en", namespace="crayola", nearest=True)) # Permanent Geranium Lake

# choose how an out-of-gamut result is resolved
for policy in (GamutPolicy.CLAMP, GamutPolicy.MAP):
    c = color_from_description("warm bright red", "en", gamut=policy)
    print(policy.value, c.hex_str)

# match against an injected, closed vocabulary
custom = ColorMatcher("xx", color_palettes=[{"#FF0000": "rood", "#0000FF": "blauw"}])
print([c.name for c, _ in custom.match_colors("maak het rood")])  # ['rood']
