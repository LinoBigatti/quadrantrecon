import os
import sys
import csv
import multiprocessing

from itertools import islice, chain, batched
from typing import List
from math import sqrt

import torch
from segment_anything import SamPredictor, sam_model_registry

import cv2
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import piexif 
import piexif.helper

from tqdm import tqdm

from .get_image_size import get_image_size

from .utils import Result, PlotUtils

BoundingBox = List[int]

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
        self.threads = 16
        self.image_width = 4000
        self.image_height = 3000
        self.cropped_width = 1700
        self.cropped_height = 1700
        self.padding_width = 45
        self.padding_height = 45
        self.horizontal_crop_left = 700
        self.horizontal_crop_right = 700
        self.vertical_crop_top = 500
        self.vertical_crop_bottom = 0
        self.log_file = open("log.txt", "w")

    def __del__(self):
        # We need to close the file handle
        self.log_file.close()

    def log(self, message: str = ""):
        if self.verbose:
            print(message)

        self.log_file.write(message + "\n")

    def imread(self, path):
        return imread_correcting_rotation(path)

    def plot_image(self, image, message: str = ""):
        if self.plot:
            self.log(f"Plotting{' ' + message if message else ''}...")

            plt.figure(figsize=(10,10))
            plt.imshow(image)

            plt.axis('on')
            plt.show()

    def get_new_filename(self, file: str, top_folder: str) -> str:
        relative_path = os.path.relpath(file, top_folder)

        new_dir = os.path.join(self.output_path, os.path.dirname(relative_path))

        new_filename = os.path.join(new_dir, os.path.basename(relative_path))

        # Create output directory first
        if not self.dry_run:
            os.makedirs(new_dir, exist_ok=True)

        return new_filename

    def get_inner_bb(self, mask, img) -> (BoundingBox, Result):
        result = Result("")

        # Remove imperfections from mask
        _mask = np.uint8(mask * 255)

        kernel = np.ones((25, 25), np.uint8)

        # Opening (erotion followed by dilation) to get rid of small noise
        _mask = cv2.morphologyEx(_mask, cv2.MORPH_OPEN, kernel)

        # Closing( dilation followed by erotion) to close up holes in bigger objects
        _mask = cv2.morphologyEx(_mask, cv2.MORPH_CLOSE, kernel)

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
        
            # Note: Lower corners are not the points closest to the image corners, but the points closest to the image corners, 500px inwards. This is because the quadrants have some handles on the sides
            for i, (x, y) in enumerate([[0, 0], [self.image_width - 500, self.image_height], [self.image_width, 0], [500, self.image_height]]):
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

        x = int(center[0] - self.cropped_width / 2)
        y = int(center[1] - self.cropped_height / 2)

        # Heuristic: Most images need to be cropped near a certain X region. Fails if its outside as a safeguard.
        MIN_X = 900
        MAX_X = 1250
        #if x < MIN_X or x > MAX_X:
            #result = result.Err("bb_cropping", "x_region_check", f"Bounding Box detection failed: Expected X between {MIN_X} and {MAX_X} (Current X: {x}, current y: {y})")

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
        
        return [x, y, x + self.cropped_width, y + self.cropped_height], result

    def main(self):
        self.create_predictor()
        
        files = {}
        _files = []
        individual_files = []

        results = []

        os.makedirs(self.output_path, exist_ok=True)
        log_table = os.path.join(self.output_path, "log.csv")
        
        if not os.path.exists(log_table):
            with open(log_table, "w", newline="") as f:
                writer = csv.writer(f, delimiter = ";")

                writer.writerow(Result.get_headers())

        # Collect filenames from all folders and subfolders listed in the arguments
        for filename in self.filename:
            if os.path.isfile(filename):
                _files.append((filename, ""))
            
            if os.path.isdir(filename):
                for root, folders, inner_files in os.walk(filename):
                    for file in inner_files:
                        if ".JPG" in file.upper() or ".JPEG" in file.upper() or ".PNG" in file.upper():
                            _files.append((os.path.join(root, file), filename))

        # Filter input files
        for file, top_folder in tqdm(_files):
            is_individual_file = not top_folder
            top_folder = top_folder if not is_individual_file else os.path.dirname(file)

            new_filename = self.get_new_filename(file, top_folder)
            relative_path = os.path.dirname(file)
            
            skip_file = False

            metadata = piexif.load(file)

            width, height = get_image_size(file)

            # Prevent running quadrantrecon on unrecognized images
            if width != self.image_width or height != self.image_height:
                self.log(f"Skipping {file}: This image has a resolution of {width}x{height} (Needed {self.image_width}x{self.image_height}).")

                skip_file = True
            
            # Prevent running quadrantrecon over already modified images
            try:
                user_comment = piexif.helper.UserComment.load(metadata["Exif"][piexif.ExifIFD.UserComment])

                if "_quadrantrecon_marker" in user_comment and not self.force:
                    self.log(f"Skipping {file}: This image has already been modified by quadrantrecon.")

                    skip_file = True
            except Exception as e:
                pass
                #self.log(f"Exception loading user comment from file {file}: {e}")

            # Prevent re-running quadrantrecon on already processed images
            if os.path.isfile(new_filename) and not self.force:
                self.log(f"Skipping {new_filename}: The output file already exists. Use --force to process it anyways.")

                skip_file = True

            if not skip_file:
                if is_individual_file:
                    individual_files.append((file, top_folder))
                else:
                    if not relative_path in files:
                        files[relative_path] = []

                    files[relative_path].append((file, top_folder))

        # Process images
        for file, top_folder in individual_files:
            result = Result(file).Err("iteration", "before_processing", "result variable was not updated")

            try:
                result = self.process_image(file, top_folder)
            except Exception as e:
                result = Result(file).Err("iteration", "after_processing", f"Exception raised: {e}")
                
                self.log(f"Encountered an error while processing file {file}: {e}")
            finally:
                results.append(result)

                with open(os.path.join(self.output_path, "log.csv"), "a", newline="") as f:
                    writer = csv.writer(f, delimiter = ";")

                    writer.writerow(result.get_as_row())

        if not files:
            return results

        # Process batch images
        iteration_c = 1
        for rel_folder in tqdm(files.keys()):
            for batched_files in batched(files[rel_folder], 200):
                result = Result(file).Err("iteration", "before_processing", "result variable was not updated")

                # Blend images
                blending_fraction = 1.0 / len(batched_files)
                blended_image = np.zeros((self.image_height, self.image_width, 3), np.uint8) 
                
                self.log(f"Loading images from path {rel_folder}... (Iteration {iteration_c})")
                iteration_c += 1

                original_images = {}

                self.log("Starting blending input images.")
                with multiprocessing.Manager() as manager:
                    # Create a tuple of arguments for each function call
                    args = [(file, blending_fraction, self.image_width, self.image_height) for file, _ in batched_files]
                    
                    with multiprocessing.Pool(self.threads) as pool:
                        # Starmap calls the function while unpacking the arguments tuple into function arguments
                        results = pool.starmap(_load_image_for_blending, args)
                       
                        pool.close()
                        pool.join()

                        # Add files together and store in cache
                        for file, img, img_blending in results:
                            np.add(blended_image, img_blending, out=blended_image)

                            original_images[file] = img

                image = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)

                self.log("Finished blending input images.")
                    
                try:
                    self.log("Images loaded.")

                    bb, result = self.process_image_no_load(image, batched_files[0][0], batched_files[0][1], True)
                    
                    for file, top_folder in batched_files:
                        original_image = original_images[file]

                        # Crop image
                        image_cropped = original_image[bb[1]:bb[3], bb[0]:bb[2]]

                        # Save image
                        if not self.dry_run and not result.failed:
                            if image_cropped.size == 0:
                                self.log(f"WARNING: Tried to write an empty image for file {file}")

                                continue

                            new_filename = self.get_new_filename(file, top_folder)

                            self.log("Saving modified image...");

                            cv2.imwrite(new_filename, image_cropped);

                            self.log("Writing metadata...")
                            
                            user_comment = piexif.helper.UserComment.dump("_quadrantrecon_marker/" + os.path.dirname(relative_path))

                            metadata["0th"][piexif.ImageIFD.XResolution] = (self.cropped_width, 1)
                            metadata["0th"][piexif.ImageIFD.YResolution] = (self.cropped_height, 1)

                            metadata["Exif"][piexif.ExifIFD.UserComment] = user_comment

                            exif_bytes = piexif.dump(metadata)
                            piexif.insert(exif_bytes, new_filename)
                except Exception as e:
                    result = Result(file).Err("iteration", "after_processing", f"Exception raised: {e}")
                    
                    self.log(f"Encountered an error while processing file {file}: {e}")
                finally:
                    results.append(result)

                    with open(os.path.join(self.output_path, "log.csv"), "a", newline="") as f:
                        writer = csv.writer(f, delimiter = ";")

                        writer.writerow(result.get_as_row())


        return results

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
        self.log(f"Loading image from path {filename}...")
        image = self.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        metadata = piexif.load(filename)
        
        self.log("Image loaded.")

        bb, result = self.process_image_no_load(image, filename, top_folder)

        # Crop image
        image_cropped = image[bb[1]:bb[3], bb[0]:bb[2]]

        # Save image
        if not self.dry_run and not result.failed:
            new_filename = self.get_new_filename(filename, top_folder)

            self.log("Saving modified image...");

            image_cropped = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR)
            cv2.imwrite(new_filename, image_cropped);

            self.log("Writing metadata...")
            
            relative_path = os.path.relpath(filename, top_folder)
            user_comment = piexif.helper.UserComment.dump("_quadrantrecon_marker/" + os.path.dirname(relative_path))

            metadata["0th"][piexif.ImageIFD.XResolution] = (self.cropped_width, 1)
            metadata["0th"][piexif.ImageIFD.YResolution] = (self.cropped_height, 1)

            metadata["Exif"][piexif.ExifIFD.UserComment] = user_comment

            exif_bytes = piexif.dump(metadata)
            piexif.insert(exif_bytes, new_filename)

        return result

    def process_image_no_load(self, image_rgb, filename: str, top_folder: str = "", batch: bool = False) -> (BoundingBox, Result):
        result = Result(filename)
        
        if not self.predictor:
            self.log("ERROR: Must load a predictor using create_predictor() first.")

            exit()
        
        new_filename = self.get_new_filename(filename, top_folder)

        if not top_folder:
            top_folder = os.path.dirname(filename)
        
        metadata = piexif.load(filename)

        self.plot_image(image_rgb, "loaded image")

        kernel = np.ones((25, 25), np.uint8)

        image_yellow = None

        if batch:
            # Create a mask to get bright pixels (Backgrounds in blended pictures are mostly dark grey)
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

            image_yellow = cv2.inRange(image_gray, 150, 255)
        else:
            bajo_amarillo = np.array([20, 50, 50], dtype=np.uint8)  # Hue 20° - High saturation and value
            alto_amarillo = np.array([60, 255, 255], dtype=np.uint8)  # Hue 60° - Max saturation and value

            # Create a mask to get yellow pixels
            image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            image_yellow = cv2.inRange(image_hsv, bajo_amarillo, alto_amarillo)

        # Clear the sides of the image, leaving only the center
        for y in range(self.image_height):
            # Crop rows
            if y < self.vertical_crop_top or y > (self.image_height - self.vertical_crop_bottom):
                image_yellow[y] = [0] * self.image_width

                continue

            # Crop columns
            if self.horizontal_crop_left > 0:
                image_yellow[y][0:self.horizontal_crop_left] = [0]

            if self.horizontal_crop_right > 0:
                image_yellow[y][-self.horizontal_crop_right:-1] = [0]

        # Opening (erotion followed by dilation) to get rid of small noise
        image_yellow = cv2.morphologyEx(image_yellow, cv2.MORPH_OPEN, kernel)

        # Closing( dilation followed by erotion) to close up holes in bigger objects
        image_yellow = cv2.morphologyEx(image_yellow, cv2.MORPH_CLOSE, kernel)

        self.plot_image(image_yellow, "yellow image")

        yellow_coords = np.column_stack(np.where(image_yellow > 0))

        random_indexes = np.random.choice(len(yellow_coords), 10, replace=False)
        random_coords = yellow_coords[random_indexes]

        self.log("Setting predictor image...")
        self.predictor.set_image(image_rgb)
        self.log("Predictor loaded")

        # Set random inner object points based on previous mask
        input_points = np.array(list([[x, y] for y, x in random_coords]))
        input_labels = np.array([1] * len(input_points))
        
        try:
            if self.plot and self.verbose:
                self.log("Plotting loaded image...")

                plt.figure(figsize=(10,10))
                plt.imshow(image_rgb)

                PlotUtils.show_points(input_points, input_labels, plt.gca())

                plt.axis('on')
                plt.show()
        except Exception as e:
            self.log(f"Exception encountered while plotting image: {e}")

        self.log("Predicting...")
        masks, scores, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )

        # The last output is the highest scoring one.
        mask = masks[-1]
        score = scores[-1]

        if self.plot and self.verbose:
            self.log("Plotting prediction...")

            plt.figure(figsize=(10, 10))
            plt.imshow(image_rgb)

            PlotUtils.show_mask(mask, plt.gca())

            plt.title(f"Predicted mask, Score: {score:.3f}", fontsize=18)
            plt.axis("off")
            plt.show()
        
        self.log("Searching for inner bounding box...")

        bb, bb_result = self.get_inner_bb(mask, image_rgb.copy())

        if self.plot:
            self.log("Plotting inner bounding box for predicted mask...")

            plt.figure(figsize=(10, 10))
            plt.imshow(image_rgb)

            PlotUtils.show_mask(mask, plt.gca())

            PlotUtils.show_box(bb, plt.gca())

            plt.title(f"Inner Bounding Box (Area to crop)", fontsize=18)
            plt.axis("off")
            plt.show()

        # Crop image
        image_cropped = image_rgb[bb[1]:bb[3], bb[0]:bb[2]]

        self.plot_image(image_cropped, "cropped image")

        if bb_result.failed:
            return bb, result.copy_from(bb_result)
        
        # If the cropped size is wrong, we ran into a corner or side.
        cropped_width = np.shape(image_cropped)[0]
        cropped_height = np.shape(image_cropped)[1]

        if cropped_width != self.cropped_width or cropped_height != self.cropped_height:
            return bb, result.Err("failsafe", "bounds_check", "Image discarded: Cropped width and height are different than expected")

        image_gray = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2GRAY)
        image_gray = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY)[1]

        kernel = np.ones((5, 5), np.uint8)
        image_gray = cv2.erode(image_gray, kernel, iterations=4)
        
        bright_content = np.count_nonzero(image_gray) / (self.cropped_width * self.cropped_height)

        MAX_BRIGHTNESS = 0.06
        if bright_content >= MAX_BRIGHTNESS:
            return bb, result.Err("failsafe", "yellow_check", f"Image discarded: Brightness/Yellow content is above {MAX_BRIGHTNESS} (Current: {bright_content})")

        return bb, result.Ok()

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

# Loads an image and multiplies it by its blending factor.
def _load_image_for_blending(file, blending_fraction, image_width, image_height):
    img = imread_correcting_rotation(file)

    # Extra failsafe for rotated/weird images
    height, width = img.shape[:2]
    if (width, height) != (image_width, image_height):
        print(f"WARNING: Skipping blending {file}: This image has a resolution of {width}x{height} (Needed {image_width}x{image_height}).")

        return

    # Convert to uint8 after each blending pass, because if we use floats the range should be from 0.0 to 1.0
    return (file, img, np.ndarray.astype(img * blending_fraction, np.uint8))
