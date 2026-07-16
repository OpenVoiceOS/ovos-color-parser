"""Color models, conversions and utility helpers."""
from ovos_color_parser import (sRGBAColor, HLSColor, color_distance, closest_color,
                               average_colors, convert_K_to_RGB,
                               get_contrasting_black_or_white, is_hex_code_valid)
from ovos_color_parser.matching import rgb_to_cmyk, cmyk_to_rgb

# models and conversions
c = sRGBAColor.from_hex_str("#1DA2DF")
print(c.as_hls, c.as_hsv, c.as_hls.as_rgb.hex_str)

# perceptual distance and closest match
red, green, blue = (sRGBAColor(255, 0, 0), sRGBAColor(0, 255, 0), sRGBAColor(0, 0, 255))
print(color_distance(red, blue))
print(closest_color(sRGBAColor(200, 30, 30), [red, green, blue]).hex_str)  # #FF0000

# averaging uses a circular hue mean
print(average_colors([HLSColor(350, 0.5, 1.0), HLSColor(10, 0.5, 1.0)]).h)  # ~0

# color temperature in Kelvin
print(convert_K_to_RGB(2700).hex_str)   # warm white
print(convert_K_to_RGB(10000).hex_str)  # cool white

# text contrast and hex validation
print(get_contrasting_black_or_white("#FFFF00").name)  # black
print(is_hex_code_valid("#abc"), is_hex_code_valid("#12345"))

# CMYK
print(rgb_to_cmyk(255, 0, 0))
print(cmyk_to_rgb(0, 100, 100, 0))
