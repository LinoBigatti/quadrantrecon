import argparse

from .qr import QuadrantRecon
from .utils import StoreMultiConstAction, DictAction, ArrayAction, get_opencv_colorspaces

import cv2

def main():
    qr = QuadrantRecon()

    parser = argparse.ArgumentParser(
        prog="QuadrantRecon",
        description="Finds and crops out quadrants in images",
        epilog="IMPORTANT: This program requires a segment anything model to be used. You can download the default one from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth and rename it to sam_vit_h.pth, or you can provide a path to a custom model with --model-path."
    )

    options_group = parser.add_argument_group("Program options")
    options_group.add_argument("-v", "--verbose",
                        help="display debug information",
                        action="store_true")
    options_group.add_argument("-p", "--plot",
                        help="plot results while working",
                        action="store_true")
    options_group.add_argument("-f", "--force",
                        help="process files even if they were already processed",
                        action="store_true")
    options_group.add_argument("--dry-run",
                        help="dont save images after cropping",
                        action="store_true")
    options_group.add_argument("--extra-metadata",
                        help="retrieve and store optional metadata from the file paths (metadata retrieved, in order: climate/date/location/site/intertidal) (default: %(default)s)",
                        action=argparse.BooleanOptionalAction,
                        default=qr.extra_metadata)
    options_group.add_argument("-j", "--threads",
                        help="number of threads to use for loading files (default: %(default)s)",
                        type=int,
                        default=qr.threads)
    options_group.add_argument("-d", "--device",
                        help="device to run on (default: %(default)s)",
                        default=qr.device)
    options_group.add_argument("--model-path",
                        help="path to the segment anything model (default: %(default)s)",
                        default=qr.model_path)
    options_group.add_argument("--model-type",
                        help="type of sam model that is being loaded (default: %(default)s)",
                        default=qr.model_type)

    input_group = parser.add_argument_group("Input images")
    input_group.add_argument("filename",
                        help="one of the images to crop",
                        nargs="+")
    input_group.add_argument("--image-width",
                        help="width of the input images, in pixels (default: %(default)i px)",
                        type=int,
                        metavar="WIDTH",
                        default=qr.image_width)
    input_group.add_argument("--image-height",
                        help="height of the input images, in pixels (default: %(default)i px)",
                        type=int,
                        metavar="HEIGHT",
                        default=qr.image_height)

    output_group = parser.add_argument_group("Output images")
    output_group.add_argument("-o", "--output-path",
                        help="folder to output cropped files to (default: %(default)s)",
                        default=qr.output_path)
    output_group.add_argument("--width",
                        help="width of the cropped area, in pixels (default: %(default)i px)",
                        type=int,
                        dest="cropped_width",
                        metavar="WIDTH",
                        default=qr.cropped_width)
    output_group.add_argument("--height",
                        help="height of the cropped area, in pixels (default: %(default)i px)",
                        type=int,
                        dest="cropped_height",
                        metavar="HEIGHT",
                        default=qr.cropped_height)
    output_group.add_argument("--padding-width",
                        help="width of the padding between the cropped area and the quadrant's top left corner, in pixels (default: %(default)i px)",
                        type=int,
                        metavar="WIDTH",
                        default=qr.padding_width)
    output_group.add_argument("--padding-height",
                        help="height of the padding between the cropped area and the quadrant's top left corner, in pixels (default: %(default)i px)",
                        type=int,
                        metavar="HEIGHT",
                        default=qr.padding_height)

    precrop_group = parser.add_argument_group("Predetection settings")
    precrop_group.add_argument("--precrop-top",
                        help="area to crop from the top before bulk processing images, in pixels (default: %(default)i px)",
                        type=int,
                        dest="vertical_crop_top",
                        metavar="AMOUNT",
                        default=qr.vertical_crop_top)
    precrop_group.add_argument("--precrop-bottom",
                        help="area to crop from the bottom before bulk processing images, in pixels (default: %(default)i px)",
                        type=int,
                        dest="vertical_crop_bottom",
                        metavar="AMOUNT",
                        default=qr.vertical_crop_bottom)
    precrop_group.add_argument("--precrop-left",
                        help="area to crop from the left before bulk processing images, in pixels (default: %(default)i px)",
                        type=int,
                        dest="horizontal_crop_left",
                        metavar="AMOUNT",
                        default=qr.horizontal_crop_left)
    precrop_group.add_argument("--precrop-right",
                        help="area to crop from the right before bulk processing images, in pixels (default: %(default)i px)",
                        type=int,
                        dest="horizontal_crop_right",
                        metavar="AMOUNT",
                        default=qr.horizontal_crop_right)
    
    color_group = parser.add_argument_group("Color detection")

    color_presets = color_group.add_mutually_exclusive_group()
    color_presets.add_argument("-b", "--predetect-brightness",
                        help="predetect the color yellow in Grayscale via brightness to pinpoint possible quadrant locations (default)",
                        action=StoreMultiConstAction,
                        const={"colorspace": cv2.COLOR_RGB2GRAY, "predetection_min_color": 150, "predetection_max_color": 255})
    color_presets.add_argument("-y", "--predetect-yellow",
                        help="predetect the color yellow in HSV space to pinpoint possible quadrant locations",
                        action=StoreMultiConstAction,
                        const={"colorspace": cv2.COLOR_RGB2HSV, "predetection_min_color": [20, 50, 50], "predetection_max_color": [60, 255, 255]})
    color_presets.add_argument("--grayscale",
                        help="use grayscale colorspace to pinpoint possible quadrant locations",
                        action="store_const",
                        dest="colorspace",
                        const=cv2.COLOR_RGB2GRAY)
    color_presets.add_argument("--colorspace", "--colourspace",
                        help="colorspace to use for pinpointing possible quadrant locations",
                        action=DictAction,
                        choices=get_opencv_colorspaces(),
                        default=qr.colorspace)

    min_color = color_group.add_mutually_exclusive_group()
    min_color.add_argument("--predetection-min-brightness",
                        help="lower brightness to use for pinpointing possible quadrant locations. Should only be used with --grayscale",
                        type=int,
                        dest="predetection_min_color",
                        metavar="BRIGHTNESS",
                        default=qr.predetection_min_color)
    min_color.add_argument("--predetection-min-color",
                        help="lower color bound to use for pinpointing possible quadrant locations. Should only be used with --colorspace",
                        action=ArrayAction,
                        metavar="COLOR_COMPONENT",
                        nargs=3)

    max_color = color_group.add_mutually_exclusive_group()
    max_color.add_argument("--predetection-max-brightness",
                        help="upper brightness to use for pinpointing possible quadrant locations. Should only be used with --grayscale",
                        type=int,
                        dest="predetection_max_color",
                        metavar="BRIGHTNESS",
                        default=qr.predetection_max_color)
    max_color.add_argument("--predetection-max-color",
                        help="upper color bound to use for pinpointing possible quadrant locations. Should only be used with --colorspace",
                        action=ArrayAction,
                        metavar="COLOR_COMPONENT",
                        nargs=3)

    parser.parse_args(namespace=qr)

    qr.main()

# The usual if __name__ == "__main__" block is omitted for __main__.py files.
# See also: https://docs.python.org/3/library/__main__.html
main()
