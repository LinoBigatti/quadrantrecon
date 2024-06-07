import os
import sys
import argparse
from itertools import islice, chain
from math import sqrt

sys.path.append("sam/")

import torch
from segment_anything import SamPredictor, sam_model_registry

import cv2
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import piexif 
import piexif.helper

class QuadrantRecon:
    def __init__(self):
        self.filename = []
        self.output_path = "./cropped_images/"
        self.verbose = False
        self.plot = False
        self.force = False
        self.dry_run = False
        self.device = "cuda"
        self.model_path = "sam_vit_h.pth"
        self.model_type = "vit_h"
        self.width = 1700
        self.height = 1700
        self.padding_width = 45
        self.padding_height = 45

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

    def log(self, message: str = ""):
        if self.verbose:
            print(message)

    def get_inner_bb(self, mask, img):
        # Detect contours in object mask
        _mask = np.uint8(mask * 255)
        contours, _hierarchy = cv2.findContours(_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area.
        # Heuristic: Largest area is gonna be the outer contour, and second largest is gonna be the inner contour.
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)

        cnt = cnts[1]
        
        # Heuristic: If the arc length is smaller than a treshold, we assume its bad
        # and grab the largest one. This is likely to be the full frame,
        # so we need more padding.
        extra_padding = 0
        if cv2.arcLength(cnt, True) < 6000:
          cnt = cnts[0]

          extra_padding = 160
        
        # Get closest point to top left corner in inner contour
        min_dist = 100000000
        min_dist_point = None
        for point in cnt:
            point = point[0]
        
            dist = sqrt(point[0] ** 2 + point[1] ** 2)
            
            if dist < min_dist:
                min_dist = dist
                min_dist_point = point
        
        if self.plot and self.verbose:
            self.log("Plotting detected corners and inner contour...")

            plt.figure(figsize=(10, 10))

            cv2.drawContours(img, cnts, -1, (255, 0, 0), 3)
            cv2.circle(img, min_dist_point, 3, (0, 255, 0), 10)

            plt.imshow(img)

            plt.axis("off")
            plt.show()


        # Add padding
        x, y = min_dist_point
        x += self.padding_width + extra_padding
        y += self.padding_height + extra_padding
        
        return [x, y, x + self.width, y + self.height] 

    def main(self):
        self.create_predictor()
        
        for filename in self.filename:
            self.process_image(filename)

    def create_predictor(self):
        if self.plot and not "google.colab" in sys.modules:
            matplotlib.use('TKagg')

        self.log("Loading model...")
        
        sam = sam_model_registry[self.model_type](checkpoint=self.model_path)
        sam.to(device = self.device)

        self.log("Model loaded.")
        
        self.log("Creating predictor...")
        self.predictor = SamPredictor(sam)

    def process_image(self, filename: str, top_folder: str = ""):
        if not self.predictor:
            print("ERROR: Must load a predictor using create_predictor() first.")

            exit()
        
        if not top_folder:
            top_folder = os.path.dirname(filename)
        
        relative_path = os.path.relpath(filename, top_folder)

        new_dir = os.path.join(self.output_path, os.path.dirname(relative_path))

        new_filename = os.path.join(new_dir, os.path.basename(relative_path))
        
        if os.path.isfile(new_filename) and not self.force:
            self.log(f"Skipping {filename}: The output file already exists. Use --force to process it anyways.")

            return

        metadata = piexif.load(filename)

        try:
            user_comment = piexif.helper.UserComment.load(metadata["Exif"][piexif.ExifIFD.UserComment])

            if "_quadrantrecon_marker" in user_comment and not self.force:
                self.log(f"Skipping {filename}: This image has already been modified by quadrantrecon.")

                return
        except ValueError:
            pass

        self.log(f"Loading image from path {filename}...")
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.log("Image loaded.")

        self.log("Setting predictor image...")
        self.predictor.set_image(image)
        self.log("Predictor loaded")

        # Set box to find objects inside
        input_box = np.array([0, 200, 4000, 3000])

        # Set inner background points to remove objects from the inside
        input_points = np.array([
            [1300, 1250], [1650, 1250], [2000, 1250], [2350, 1250], [2700, 1250],
            [1300, 1500], [1650, 1500], [2000, 1500], [2350, 1500], [2700, 1500],
            [1300, 1750], [1650, 1750], [2000, 1750], [2350, 1750], [2700, 1750],
            [1300, 2000], [1650, 2000], [2000, 2000], [2350, 2000], [2700, 2000],
            [1300, 2250], [1650, 2250], [2000, 2250], [2350, 2250], [2700, 2250],
        ])
        input_labels = np.array([0] * len(input_points))
        
        if self.plot and self.verbose:
            self.log("Plotting loaded image...")

            plt.figure(figsize=(10,10))
            plt.imshow(image)
            self.show_box(input_box, plt.gca())
            self.show_points(input_points, input_labels, plt.gca())
            plt.axis('on')
            plt.show()

        self.log("Predicting...")
        masks, scores, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_box[None, :],
            multimask_output=True,
        )

        # The last output is the highest scoring one.
        mask = masks[-1]
        score = scores[-1]

        if self.plot and self.verbose:
            self.log("Plotting prediction...")

            plt.figure(figsize=(10, 10))
            plt.imshow(image)

            self.show_mask(mask, plt.gca())

            plt.title(f"Predicted mask, Score: {score:.3f}", fontsize=18)
            plt.axis("off")
            plt.show()
        
        self.log("Searching for inner bounding box...")

        bb = self.get_inner_bb(mask, image.copy())

        if self.plot:
            self.log("Plotting inner bounding box for predicted mask...")

            plt.figure(figsize=(10, 10))
            plt.imshow(image)

            self.show_mask(mask, plt.gca())

            self.show_box(bb, plt.gca())

            plt.title(f"Inner Bounding Box (Area to crop)", fontsize=18)
            plt.axis("off")
            plt.show()

        # Crop image
        image_cropped = image[bb[1]:bb[3], bb[0]:bb[2]]

        if self.plot and self.verbose:
            plt.figure(figsize=(10, 10))
            plt.imshow(image_cropped)
            plt.title(f"Cropped Image", fontsize=18)
            plt.axis("off")
            plt.show()  

        failed = False
        
        # If the cropped size is wrong, we ran into a corner or side.
        cropped_width = np.shape(image_cropped)[0]
        cropped_height = np.shape(image_cropped)[1]

        if cropped_width != self.width or cropped_height != self.height:
          failed = True

        # Detect yellow color in image.
        if not failed:
            image_hsv = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2HSV)
            lower = np.array([22, 93, 0], dtype="uint8")
            upper = np.array([45, 255, 255], dtype="uint8")

            yellow_mask = cv2.inRange(image, lower, upper)
            yellow_content = np.count_nonzero(yellow_mask) / (self.width * self.height)
        
            # Heuristic: If yellow content in the image is too high, we probably
            # cropped a part of the quadrant.
            if yellow_content >= 0.012:
              failed = True

              self.log(f"{yellow_content:}")

        # Save image
        if not self.dry_run and not failed:
            self.log("Saving modified image...");

            os.makedirs(new_dir, exist_ok=True)

            if not top_folder:
                top_folder = os.path.dirname(filename)
            
            relative_path = os.path.relpath(filename, top_folder)

            new_dir = os.path.join(self.output_path, os.path.dirname(relative_path))
            os.makedirs(new_dir, exist_ok=True)

            new_filename = os.path.join(new_dir, os.path.basename(relative_path))

            image_cropped = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR)
            cv2.imwrite(new_filename, image_cropped);

            self.log("Writing metadata...")
            
            user_comment = piexif.helper.UserComment.dump("_quadrantrecon_marker/" + os.path.dirname(relative_path))

            metadata["0th"][piexif.ImageIFD.XResolution] = (self.width, 1)
            metadata["0th"][piexif.ImageIFD.YResolution] = (self.height, 1)

            metadata["Exif"][piexif.ExifIFD.UserComment] = user_comment

            exif_bytes = piexif.dump(metadata)
            piexif.insert(exif_bytes, new_filename)

        return failed

if __name__ == "__main__":
    qr = QuadrantRecon()

    parser = argparse.ArgumentParser(
        prog="QuadrantRecon",
        description="Finds and crops out quadrants in images",
    )

    parser.add_argument("filename",
                        help="one of the images to crop",
                        nargs="+")
    parser.add_argument("-o", "--output_path",
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
    parser.add_argument("--width",
                        help="width of the cropped area, in pixels (default: %(default)ipx)",
                        type=int,
                        default=qr.width)
    parser.add_argument("--height",
                        help="height of the cropped area, in pixels (default: %(default)ipx)",
                        type=int,
                        default=qr.height)
    parser.add_argument("--padding_width",
                        help="width of the padding between the cropped area and the quadrant's top left corner, in pixels (default: %(default)ipx)",
                        type=int,
                        default=qr.padding_width)
    parser.add_argument("--padding_height",
                        help="height of the padding between the cropped area and the quadrant's top left corner, in pixels (default: %(default)ipx)",
                        type=int,
                        default=qr.padding_height)

    parser.parse_args(namespace=qr)

    qr.main()

