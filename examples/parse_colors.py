"""Parse natural-language color descriptions into color objects."""
from ovos_color_parser import color_from_description

for description in ["red", "dark red", "light blue", "vivid green",
                    "warm gray", "transparent yellow", "navy blue"]:
    color = color_from_description(description, lang="en")
    print(f"{description!r:25} -> {color.hex_str} {color}")

# unknown descriptions return None
print(color_from_description("qzxwv", lang="en"))

# force a known, named color from the matched candidates
color = color_from_description("red", lang="en", cast_to_palette=True)
print(f"cast_to_palette: {color.name} {color.hex_str}")
