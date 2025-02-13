import argparse

import cv2
import numpy as np

import piexif 
import piexif.helper

import matplotlib
import matplotlib.pyplot as plt

class Result:
    def __init__(self, filename: str = ""):
        self.failed = False
        self.filename = filename
        self.process = ""
        self.current_block = ""
        self.error_reason = ""
    
    def Err(self, process: str, block: str, reason: str):
        self.failed = True
        self.process = process
        self.current_block = block
        self.error_reason = reason

        return self

    def Ok(self):
        self.failed = False

        return self

    def copy_from(self, other):
        self.failed = other.failed 
        self.process = other.process
        self.current_block = other.current_block
        self.error_reason = other.error_reason

        return self

    def get_as_row(self):
        return [self.failed, self.filename, self.process, self.current_block, self.error_reason]

    def get_headers():
        return ["Failed", "Filename", "Process", "Execution Block", "Error"]

    def __str__(self):
        return f"{self.filename}: {'Err: ' + self.error_reason if self.failed else 'Ok'}"


class PlotUtils:
    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])

        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

class StoreMultiConstAction(argparse.Action):
    def __init__(self,
                 option_strings,
                 dest,
                 const=None,
                 default=None,
                 required=False,
                 help=None,
                 metavar=None,
                 deprecated=False):
        super(StoreMultiConstAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            const=const,
            default=default,
            required=required,
            help=help,
            deprecated=deprecated)

    def __call__(self, parser, namespace, values, option_string=None):
        for dst, const in self.const.items():
            setattr(namespace, dst, const)

    def format_usage(self):
        return ' | '.join(self.option_strings)

class DictAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.choices.get(values, self.default))

class ArrayAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        _values = np.array(values, dtype=np.uint8)

        setattr(namespace, self.dest, _values)

def get_opencv_colorspaces():
    return {
        "bgr": cv2.COLOR_RGB2BGR,
        "ycc": cv2.COLOR_RGB2YCrCb,
        "yuv": cv2.COLOR_RGB2YUV,
        "hsv": cv2.COLOR_RGB2HSV,
        "hsv_full": cv2.COLOR_RGB2HSV_FULL,
        "hls": cv2.COLOR_RGB2HLS,
        "hls_full": cv2.COLOR_RGB2HLS_FULL,
        "ciexyz": cv2.COLOR_RGB2XYZ,
        "cielab": cv2.COLOR_RGB2Lab,
        "cieluv": cv2.COLOR_RGB2Luv,
    }

def imread_correcting_rotation(path):
    img = cv2.imread(path)
    metadata = piexif.load(path)

    if not piexif.ImageIFD.Orientation in metadata["0th"]:
        return img

    # Correct image orientation before opening.
    orientation = metadata["0th"][piexif.ImageIFD.Orientation]
    
    if orientation == 2:
        img = cv2.flip(img, 1)
    elif orientation == 3:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif orientation == 4:
        img = cv2.rotate(img, cv2.ROTATE_180)
        img = cv2.flip(img, 1)
    elif orientation == 5:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.flip(img, 1)
    elif orientation == 6:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 7:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.flip(img, 1)
    elif orientation == 8:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    return img

