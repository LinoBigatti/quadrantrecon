import sys
import argparse
from itertools import islice, chain

import torch
from segment_anything import SamPredictor, sam_model_registry

import cv2
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import piexif 
import piexif.helper

class QuadrantRecon:
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

    def get_inner_bb(self, mask):
        # We build a version with the axes flipped to prevent an O(n2) situation in the second loop 
        flipped_mask = np.swapaxes(mask, 0, 1)

        start_y = -1
        end_y = -1
        
        prev_had_mask = False
        for i, row in enumerate(mask):
            has_mask = sum(row) > 500
            
            if not has_mask and prev_had_mask and start_y == -1:
                start_y = i
            
            if has_mask and not prev_had_mask and start_y != -1:
                if end_y == 0:
                    end_y = i - 1
                elif end_y == -1:
                    end_y += 1

            prev_had_mask = has_mask
        
        start_x = -1
        end_x = -1

        prev_had_mask = False
        for i, col in enumerate(flipped_mask):
            has_mask = sum(col) > 700

            if not has_mask and prev_had_mask and start_x == -1:
                start_x = i

            if has_mask and not prev_had_mask and start_x != -1 and end_x == -1:
                end_x = i - 1

            prev_had_mask = has_mask

        cropped_mask = [list(islice(row, start_x, end_x)) for row in islice(mask, start_y, end_y)]
        flipped_cropped_mask = np.swapaxes(cropped_mask, 0, 1)
        
        bb_x = 0
        bb_y = 0
        
        done = False
        while not done:
            done = True

            # Check row
            if True in cropped_mask[bb_y][bb_x:bb_x+self.width]:
                bb_y += 1
                
                done = False

            # Check column
            if True in flipped_cropped_mask[bb_x][bb_y:bb_y+self.height]:
                bb_x += 1

                done = False

        x = start_x + bb_x
        y = start_y + bb_y

        return [x, y, x + self.width, y + self.height] 

    def main(self):
        if self.plot:
            matplotlib.use('TKagg')

        self.log("Loading model...")
        
        sam = sam_model_registry[self.model_type](checkpoint=self.model_path)
        sam.to(device = self.device)

        self.log("Model loaded.")
        
        self.log("Creating predictor...")
        predictor = SamPredictor(sam)

        for filename in self.filename:
            self.process_image(filename, predictor)

    def process_image(self, filename: str, predictor: SamPredictor):
        metadata = piexif.load(filename)

        try:
            user_comment = piexif.helper.UserComment.load(metadata["Exif"][piexif.ExifIFD.UserComment])

            if user_comment == "_quadrantrecon_marker":
                print(f"Skipping {filename}: This image has already been modified by quadrantrecon.")

                return
        except ValueError:
            pass

        self.log(f"Loading image from path ${filename}...")
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.log("Image loaded.")

        self.log("Setting predictor image...")
        predictor.set_image(image)
        self.log("Predictor loaded")

        input_points = np.array([[1000, 650], [3000, 650], [1000, 2650], [3000, 2650]])
        input_labels = np.array([1, 1, 1, 1])
        
        if self.plot and self.verbose:
            self.log("Plotting loaded image...")

            plt.figure(figsize=(10,10))
            plt.imshow(image)
            self.show_points(input_points, input_labels, plt.gca())
            plt.axis('on')
            plt.show()

        self.log("Predicting...")
        masks, scores, _ = predictor.predict(
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
            plt.imshow(image)

            self.show_mask(mask, plt.gca())

            plt.title(f"Predicted mask, Score: {score:.3f}", fontsize=18)
            plt.axis("off")
            plt.show()
        
        self.log("Searching for inner bounding box...")

        bb = self.get_inner_bb(mask)

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

        # Save image
        self.log("Saving modified image...");
        
        new_filename = filename[:-4] + "_cropped.JPG"

        cv2.imwrite(new_filename, image_cropped);

        self.log("Writing metadata...")

        metadata["0th"][piexif.ImageIFD.XResolution] = (self.width, 1)
        metadata["0th"][piexif.ImageIFD.YResolution] = (self.height, 1)

        user_comment = piexif.helper.UserComment.dump(u"_quadrantrecon_marker")
        metadata["Exif"][piexif.ExifIFD.UserComment] = user_comment

        exif_bytes = piexif.dump(metadata)
        piexif.insert(exif_bytes, new_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="QuadrantRecon",
        description="Finds and crops out quadrants in images",
    )

    parser.add_argument("filename",
                        help="one of the images to crop",
                        nargs="+")
    parser.add_argument("-v", "--verbose",
                        help="display debug information",
                        action="store_true")
    parser.add_argument("-p", "--plot",
                        help="plot results while working",
                        action="store_true")
    parser.add_argument("-d", "--device",
                        help="device to run on (default: %(default)s)",
                        default="cuda")
    parser.add_argument("--width",
                        help="width of the cropped area, in pixels (default: %(default)ipx)",
                        type=int,
                        default=1790)
    parser.add_argument("--height",
                        help="height of the cropped area, in pixels (default: %(default)ipx)",
                        type=int,
                        default=1790)
    parser.add_argument("--model-path",
                        help="path to the segment anything model (default: %(default)s)",
                        default="sam_vit_h.pth")
    parser.add_argument("--model-type",
                        help="type of sam model that is being loaded (default: %(default)s)",
                        default="vit_h")


    qr = QuadrantRecon()
    parser.parse_args(namespace=qr)

    qr.main()
