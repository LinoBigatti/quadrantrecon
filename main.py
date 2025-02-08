import argparse

from qr import QuadrantRecon

if __name__ == "__main__":
    qr = QuadrantRecon()

    parser = argparse.ArgumentParser(
        prog="QuadrantRecon",
        description="Finds and crops out quadrants in images",
    )

    parser.add_argument("filename",
                        help="one of the images to crop",
                        nargs="+")
    parser.add_argument("-o", "--output-path",
                        help="folder to output cropped files to (default: %(default)s)",
                        default=qr.output_path)
    parser.add_argument("-v", "--verbose",
                        help="display debug information",
                        action="store_true")
    parser.add_argument("-p", "--plot",
                        help="plot results while working",
                        action="store_true")
    parser.add_argument("-f", "--force",
                        help="process files even if they were already processed",
                        action="store_true")
    parser.add_argument("--dry-run",
                        help="dont save images after cropping",
                        action="store_true")
    parser.add_argument("-d", "--device",
                        help="device to run on (default: %(default)s)",
                        default=qr.device)
    parser.add_argument("--model-path",
                        help="path to the segment anything model (default: %(default)s)",
                        default=qr.model_path)
    parser.add_argument("--model-type",
                        help="type of sam model that is being loaded (default: %(default)s)",
                        default=qr.model_type)
    parser.add_argument("-j", "--threads",
                        help="number of threads to use for loading files (default: %(default))",
                        type=int,
                        default=qr.threads)
    parser.add_argument("--width",
                        help="width of the cropped area, in pixels (default: %(default)ipx)",
                        type=int,
                        dest="cropped_width",
                        default=qr.cropped_width)
    parser.add_argument("--height",
                        help="height of the cropped area, in pixels (default: %(default)ipx)",
                        type=int,
                        dest="cropped_height",
                        default=qr.cropped_height)
    parser.add_argument("--image-width",
                        help="width of the input images, in pixels (default: %(default)ipx)",
                        type=int,
                        default=qr.image_width)
    parser.add_argument("--image-height",
                        help="height of the input images, in pixels (default: %(default)ipx)",
                        type=int,
                        default=qr.image_height)
    parser.add_argument("--padding-width",
                        help="width of the padding between the cropped area and the quadrant's top left corner, in pixels (default: %(default)ipx)",
                        type=int,
                        default=qr.padding_width)
    parser.add_argument("--padding-height",
                        help="height of the padding between the cropped area and the quadrant's top left corner, in pixels (default: %(default)ipx)",
                        type=int,
                        default=qr.padding_height)
    parser.add_argument("--precrop-top",
                        help="area to crop from the top before bulk processing images, in pixels (default: %(default)ipx)",
                        type=int,
                        dest="vertical_crop_top",
                        default=qr.vertical_crop_top)
    parser.add_argument("--precrop-bottom",
                        help="area to crop from the bottom before bulk processing images, in pixels (default: %(default)ipx)",
                        type=int,
                        dest="vertical_crop_bottom",
                        default=qr.vertical_crop_bottom)
    parser.add_argument("--precrop-left",
                        help="area to crop from the left before bulk processing images, in pixels (default: %(default)ipx)",
                        type=int,
                        dest="horizontal_crop_left",
                        default=qr.horizontal_crop_left)
    parser.add_argument("--precrop-right",
                        help="area to crop from the right before bulk processing images, in pixels (default: %(default)ipx)",
                        type=int,
                        dest="horizontal_crop_right",
                        default=qr.horizontal_crop_right)

    parser.parse_args(namespace=qr)

    qr.main()

