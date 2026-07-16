# Color description semantics

When describing color in natural language to approximate it in RGB, there are several keywords that can convey
its properties effectively

```python
# Parse complex color descriptions
color = color_from_description("very bright, slightly warm muted blue")
```

<details>
  <summary>Hue</summary>

- **Description**: Hue refers to the basic color family, such as red, blue, green, or yellow.
- **Translation to RGB**:
    - The hue determines which of the primary RGB channels (red, green, or blue) will be most prominent. For example,
      “red” means a strong red channel with low green and blue, while “blue” means a high blue channel with low red and
      green.
    - Hues like "yellow" indicate both red and green channels are high with blue low, while "purple" combines red and
      blue with little green.

</details>

<details>
  <summary>Saturation (Intensity)</summary>

- **Description**: Saturation, or chroma, is how pure or intense the color is. Terms like “vibrant,” “dull,” or “washed
  out” refer to saturation.
- **Translation to RGB**:
    - High saturation (vibrant): Increase the difference between the dominant channel(s) and others. For example, making
      the red channel much higher than green and blue for a vibrant red.
    - Low saturation (dull): Reduce the contrast between channels, creating a blend closer to grayscale. For instance,
      balancing red, green, and blue channels to similar values lowers saturation.

</details>

<details>
  <summary>Brightness (Value or Lightness)</summary>

- **Description**: Brightness refers to how light or dark the color appears. Words like “bright,” “dim,” “dark,” or
  “pale” are often used.
- **Translation to RGB**:
    - High brightness (bright): Increase the values across all channels.
    - Low brightness (dark): Decrease values across channels while maintaining the hue's relative balance.

</details>


<details>
  <summary>Temperature (Warmth/Coolness)</summary>

- **Description**: Color temperature reflects whether a color feels warm or cool. Terms like "warm red," "cool green,"
  or "cold blue" apply here.
- **Translation to RGB**:
    - Warm colors: Increase red or red and green channels.
    - Cool colors: Increase blue or decrease red.

</details>

<details>
  <summary>Opacity/Transparency</summary>

- **Description**: Opacity doesn’t affect RGB but is relevant for color perception, especially in design. Terms like
  “translucent,” “opaque,” or “sheer” describe it.
- **Translation to RGB**:
    - Opacity affects the alpha channel (RGBA) rather than RGB values.

</details>

This approach, while interpretative, offers a structured way to translate natural language color descriptions into RGB
approximations.

#### Color Keywords 


To categorize adjectives and keywords that describe color in ways that translate into RGB or color space adjustments the
parser uses a `.json` file per language

Example JSON structure for English color keywords:
 
```json
{
  "saturation": {
    "high": ["vibrant", "rich", "bold", "deep"],
    "low": ["dull", "muted", "washed-out", "faded"]
  },
  "brightness": {
    "high": ["bright", "light", "pale", "glowing"],
    "low": ["dim", "dark", "shadowy", "faint"]
  }
}
```

Color name lists in each language are also used to determine the **hue**. 

> English has a word list of almost ~6000 color name mappings


Below are some examples of non-color-name keywords that define other qualities of a color

<details>
  <summary>Saturation (Intensity)</summary>

- **Very High Saturation**: For colors that are extremely intense or vivid.
    - Keywords: “neon,” “saturated,” “intense,” “brilliant,” “flamboyant”
- **High Saturation**: These adjectives indicate vibrant or intense colors where the hue is pronounced.
    - Keywords: “vibrant,” “rich,” “bold,” “deep,” “vivid,” “intense,” “pure,” “electric”
- **Low Saturation**: These adjectives imply a muted or washed-out appearance, often making the color appear closer to
  grayscale.
    - Keywords: “dull,” “muted,” “washed-out,” “faded,” “soft,” “pale,” “subdued,” “pastel”
- **Very Low Saturation**: For colors that are very desaturated, nearing grayscale.
    - Keywords: “drab,” “grayed,” “washed-out,” “faded,” “subdued”
</details>

<details>
  <summary>Brightness (Lightness/Value)</summary>

- **Very High Brightness**: Extremely bright colors, often implying high lightness or near-whiteness.
    - Keywords: “blinding,” “radiant,” “glowing,” “white,” “light-filled”
- **High Brightness**: Bright colors, often indicating a lighter shade or close to white.
    - Keywords: “bright,” “light,” “pale,” “glowing,” “luminous,” “brilliant,” “clear,” “radiant”
- **Low Brightness**: These terms describe darker or dimmer shades, closer to black.
    - Keywords: “dim,” “dark,” “shadowy,” “faint,” “gloomy,” “subdued,” “deep,” “midnight”
- **Very Low Brightness**: Colors that are nearly black or very dark.
    - Keywords: “pitch-dark,” “black,” “shadowed,” “deep,” “ink-like”

</details>

<details>
  <summary>Temperature (Warmth)</summary>

- **Very High Temperature (Very Warm)**: Intense warm colors, strongly leaning toward red, orange, or intense yellow.
    - Keywords: “fiery,” “lava-like,” “burning,” “blazing”
- **High Temperature (Warm Colors)**: Warmer colors suggest a shift towards red or yellow tones, giving the color a
  warmer feel.
    - Keywords: “warm,” “hot,” “fiery,” “sunny,” “toasty,” “scorching,” “amber,” “reddish”
- **Low Temperature (Cool Colors)**: Cooler colors involve blue or green tones, giving the color a cooler or icy
  appearance.
    - Keywords: “cool,” “cold,” “chilly,” “icy,” “frosty,” “crisp,” “bluish,” “aqua”
- **Very Low Temperature (Very Cool)**: Extremely cool tones, verging on cold, icy blues or greens.
    - Keywords: “icy,” “arctic,” “frigid,” “wintry,” “glacial”
</details>

<details>
  <summary>Opacity/Transparency</summary>


- **Very High Opacity**: Extremely solid or dense colors.
    - Keywords: “impenetrable,” “opaque,” “thick”
- **High Opacity**: Describes solid colors without transparency.
    - Keywords: “opaque,” “solid,” “dense,” “thick,” “cloudy,” “impenetrable,” “strong”
- **Low Opacity**: Indicates transparency or translucency, where the background may show through.
    - Keywords: “transparent,” “translucent,” “sheer,” “see-through,” “misty,” “delicate,” “airy”
- **Very Low Opacity**: Highly transparent or barely visible colors.
    - Keywords: “ethereal,” “ghostly,” “barely-there,” “translucent”

</details>


