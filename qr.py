import os
import sys
import csv

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

from tqdm import tqdm

from utils import Result, PlotUtils

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

    def log(self, message: str = ""):
        if self.verbose:
            print(message)

    def get_inner_bb(self, mask, img):
        result = Result("")

        # Remove imperfections from mask
        _mask = np.uint8(mask * 255)

        kernel = np.ones((25, 25), np.uint8)
        _mask = cv2.erode(_mask, kernel, iterations=1)

        # Detect contours in object mask
        contours, _hierarchy = cv2.findContours(_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area.
        # Heuristic: Largest area is gonna be the outer contour, and second largest is gonna be the inner contour.
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)

        if not cnts:
            return [0, 0, 0, 0], result.Err("bb_cropping", "contour_detection", "Bounding Box detection failed: No contours were detected.")

        cnt = cnts[0]
        
        # Heuristic: If the contour length is too small, its probably not the right object.
        MIN_CNT_LEN = 6000
        if a := cv2.arcLength(cnt, True) < MIN_CNT_LEN:
            result = result.Err("bb_cropping", "arc_length_check", f"Bounding Box detection failed: Expected contour arc length to be above {MIN_CNT_LEN} (current: {a})")
        
        # Get closest points to corners in contour
        corners = [None] * 4
        corner_dists = [100000000] * 4
        for point in cnt:
            point = point[0]
        
            # Note: Lower corners are not the image corners, but some points 500px inwards. This is because the quadrants have some handles on the sides
            for i, (x, y) in enumerate([[0, 0], [3500, 3000], [4000, 0], [500, 3000]]):
                # Euclidean distance
                dist = sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
                
                if dist < corner_dists[i]:
                    corners[i] = point
                    corner_dists[i] = dist

        # Get the intersection between the lines formed by the opposing corner points. We will use that as the center of the quadrant.
        # TODO: Document algorithm.
        da = corners[1] - corners[0]
        db = corners[3] - corners[2]
        dp = corners[0] - corners[2]
        dap = np.empty_like(da)
        dap[0] = -da[1]
        dap[1] = da[0]

        denom = np.dot(dap, db)
        num = np.dot(dap, dp)
        center = np.ndarray.astype((num / denom) * db + corners[2], np.uint32)

        x = int(center[0] - self.width / 2)
        y = int(center[1] - self.height / 2)

        # Heuristic: Most images need to be cropped near a certain X region. Fails if its outside as a safeguard.
        MIN_X = 900
        MAX_X = 1200
        if x < MIN_X or x > MAX_X:
            result = result.Err("bb_cropping", "x_region_check", f"Bounding Box detection failed: Expected X between {MIN_X} and {MAX_X} (Current X: {x}, current y: {y})")

        if self.plot and self.verbose:
            self.log("Plotting detected corners and inner contour...")

            plt.figure(figsize=(10, 10))

            cv2.drawContours(img, cnts, -1, (255, 0, 0), 3)
            
            for p in corners:
                cv2.circle(img, p, 3, (0, 255, 0), 10)

            cv2.circle(img, center, 7, (255, 0, 255), 10)

            plt.imshow(img)

            plt.axis("off")
            plt.show()

        if not result.failed:
            result = result.Ok()
        
        return [x, y, x + self.width, y + self.height], result

    def main(self):
        self.create_predictor()
        
        files = []

        os.makedirs(self.output_path, exist_ok=True)
        with open(os.path.join(self.output_path, "log.csv"), "w", newline="") as f:
            writer = csv.writer(f, delimiter = ";")

            writer.writerow(Result.get_headers())

        for filename in self.filename:
            if os.path.isfile(filename):
                files.append((filename, ""))
            
            if os.path.isdir(filename):
                for root, folders, _files in os.walk(filename):
                    for file in _files:
                        if ".JPG" in file.upper() or ".JPEG" in file.upper() or ".PNG" in file.upper():
                            files.append((os.path.join(root, file), filename))

        for file, top_folder in tqdm(files, disable=self.plot):
            result = Result(file).Err("iteration", "before_processing", "result variable was not updated")

            try:
                result = self.process_image(file, top_folder)
            except Exception as e:
                result = Result(file).Err("iteration", "after_processing", f"Exception raised: {e}")
            finally:
                with open(os.path.join(self.output_path, "log.csv"), "a", newline="") as f:
                    writer = csv.writer(f, delimiter = ";")

                    writer.writerow(result.get_as_row())

    def create_predictor(self):
        if self.plot and not "google.colab" in sys.modules:
            matplotlib.use('TKagg')

        self.log("Loading model...")
        
        sam = sam_model_registry[self.model_type](checkpoint=self.model_path)
        sam.to(device = self.device)

        self.log("Model loaded.")
        
        self.log("Creating predictor...")
        self.predictor = SamPredictor(sam)

    def process_image(self, filename: str, top_folder: str = "") -> Result:
        result = Result(filename)

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

            return result.Err("preprocessing", "check_output_file", "File skipped: Output file already exists")

        metadata = piexif.load(filename)

        try:
            user_comment = piexif.helper.UserComment.load(metadata["Exif"][piexif.ExifIFD.UserComment])

            if "_quadrantrecon_marker" in user_comment and not self.force:
                self.log(f"Skipping {filename}: This image has already been modified by quadrantrecon.")

                return result.Err("preprocessing", "check_input_file", "File skipped: Input file was marked by quadrantrecon")
        except:
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
        
        try:
            if self.plot and self.verbose:
                self.log("Plotting loaded image...")

                plt.figure(figsize=(10,10))
                plt.imshow(image)

                PlotUtils.show_box(input_box, plt.gca())

                PlotUtils.show_points(input_points, input_labels, plt.gca())

                plt.axis('on')
                plt.show()
        except Exception as e:
            print(e)

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

            PlotUtils.show_mask(mask, plt.gca())

            plt.title(f"Predicted mask, Score: {score:.3f}", fontsize=18)
            plt.axis("off")
            plt.show()
        
        self.log("Searching for inner bounding box...")

        bb, bb_result = self.get_inner_bb(mask, image.copy())

        if self.plot:
            self.log("Plotting inner bounding box for predicted mask...")

            plt.figure(figsize=(10, 10))
            plt.imshow(image)

            PlotUtils.show_mask(mask, plt.gca())

            PlotUtils.show_box(bb, plt.gca())

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

        if bb_result.failed:
            return result.copy_from(bb_result)
        
        # If the cropped size is wrong, we ran into a corner or side.
        cropped_width = np.shape(image_cropped)[0]
        cropped_height = np.shape(image_cropped)[1]

        if cropped_width != self.width or cropped_height != self.height:
            return result.Err("failsafe", "bounds_check", "Image discarded: Cropped width and height are different than expected")

        image_gray = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2GRAY)
        image_gray = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY)[1]

        kernel = np.ones((5, 5), np.uint8)
        image_gray = cv2.erode(image_gray, kernel, iterations=4)
        
        bright_content = np.count_nonzero(image_gray) / (self.width * self.height)

        MAX_BRIGHTNESS = 0.01
        if bright_content >= MAX_BRIGHTNESS:
            return result.Err("failsafe", "yellow_check", f"Image discarded: Brightness/Yellow content is above {MAX_BRIGHTNESS} (Current: {bright_content})")

        # Save image
        if not self.dry_run:
            self.log("Saving modified image...");

            os.makedirs(new_dir, exist_ok=True)

            image_cropped = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR)
            cv2.imwrite(new_filename, image_cropped);

            self.log("Writing metadata...")
            
            user_comment = piexif.helper.UserComment.dump("_quadrantrecon_marker/" + os.path.dirname(relative_path))

            metadata["0th"][piexif.ImageIFD.XResolution] = (self.width, 1)
            metadata["0th"][piexif.ImageIFD.YResolution] = (self.height, 1)

            metadata["Exif"][piexif.ExifIFD.UserComment] = user_comment

            exif_bytes = piexif.dump(metadata)
            piexif.insert(exif_bytes, new_filename)

        return result.Ok()
