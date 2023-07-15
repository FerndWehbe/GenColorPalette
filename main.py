from PIL import Image, ImageDraw
import colorsys
import numpy
import rich
import sys


HEIGHT = 16


def rgb_to_hsl(rgb_colors):
    rgb_colors = rgb_colors / 255.0
    return numpy.array(
        [colorsys.rgb_to_hls(*rgb_color[:3]) for rgb_color in rgb_colors]
    )


def extract_palette(image_path):
    img = Image.open(image_path)
    array_img = numpy.array(img)
    image_channels = array_img.shape[2]
    array_unidimensional = array_img.reshape(-1, image_channels)

    if image_channels == 4:
        alpha_channel = array_unidimensional[:, 3]
        non_zero_alpha_indices = alpha_channel.nonzero()[0]
        array_unidimensional = array_unidimensional[non_zero_alpha_indices]

    colors = numpy.unique(array_unidimensional, axis=0)
    return colors


def create_palette(color_array: numpy.ndarray, palette_name):
    hsl_colors = rgb_to_hsl(color_array)
    mode = "RGB"
    if color_array.shape[1] == 4:
        mode = "RGBA"
        hsl_colors = numpy.concatenate(
            (hsl_colors, color_array[:, 3:4]), axis=1
        )

    hue = hsl_colors[:, 0]
    sorted_indices = numpy.argsort(hue)
    sorted_colors = color_array[sorted_indices]

    new_image = Image.new(mode, (HEIGHT * color_array.shape[0], HEIGHT))
    for index, color in enumerate(sorted_colors):
        rect = [(index * HEIGHT, 0), ((index * HEIGHT) + HEIGHT, HEIGHT)]
        ImageDraw.Draw(new_image).rectangle(rect, fill=tuple(color))

    if palette_name:
        new_image.save(palette_name)
        return

    new_image.save("palette.png")


def execute_palette_generation(image_path, palette_name):
    color_array = extract_palette(image_path)
    create_palette(color_array, palette_name)


def main():
    if len(sys.argv) < 2:
        rich.print("[bold red]Error: Image path not provided.[/bold red]")
        rich.print("Usage: ./palette_generator.py <image_path> [palette_name]")
        return

    image_path = sys.argv[1]
    palette_name = sys.argv[2] if len(sys.argv) >= 3 else None

    execute_palette_generation(image_path, palette_name)
    rich.print(
        f"Color palette generated from the image '{image_path}'"
        f" saved with name '{palette_name if palette_name else 'palette.png'}'"
    )


if __name__ == "__main__":
    main()
