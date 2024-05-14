import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import pyexiv2
from segment_anything import SamPredictor, sam_model_registry
from PyQt5 import QtWidgets
from itertools import islice, chain

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

__DEBUG__: bool = False
def _log(message: str = ""):
    if __DEBUG__:
        print(message)

def get_inner_bb(mask):
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

    while sum(chain.from_iterable(cropped_mask)) > 0:
        print(sum(chain.from_iterable(cropped_mask)))
        start_x += 1
        end_x -= 1

        start_y += 1
        end_y -= 1

        cropped_mask = [list(islice(row, start_x, end_x)) for row in islice(mask, start_y, end_y)]

    return [start_x, start_y, end_x, end_y] 

def main(filename: str, _plot: bool):
    _log(f"Loading image from path ${filename}...")
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    metadata = pyexiv2.ImageMetadata(filename)
    metadata.read()

    if metadata['Exif.Photo.UserComment'].raw_value == "_quadrantrecon_marker":
        print("ERROR: This image has already been modified by quadrantrecon.")

        exit()

    _log("Image loaded.")

    _log("Loading model...")
    
    sam = sam_model_registry["default"](checkpoint="./models/sam_vit_h_4b8939.pth")
    sam.to(device = "cuda")

    _log("Model loaded.")
    
    _log("Creating predictor...")
    predictor = SamPredictor(sam)
    _log("Setting predictor image...")
    predictor.set_image(image)
    _log("Predictor loaded")

    input_points = np.array([[1000, 650], [3000, 650], [1000, 2650], [3000, 2650]])
    input_labels = np.array([1, 1, 1, 1])
    
    if _plot:
        _log("Plotting loaded image...")

        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_points(input_points, input_labels, plt.gca())
        plt.axis('on')
        plt.show()

    _log("Predicting...")
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )

    max_score = -1
    max_score_mask = None

    _log("Showing predictions")
    for i, (mask, score) in enumerate(zip(masks, scores)):
        if score > max_score:
            max_score_mask = mask
        
        if __DEBUG__ and _plot:
            _log(f"Plotting prediction ${i}...")
        
            plt.figure(figsize=(10,10))
            plt.imshow(image)

            show_mask(mask, plt.gca())
            show_points(input_points, input_labels, plt.gca())

            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis("off")
            plt.show()


    if _plot:
        _log("Plotting biggest score prediction...")

        plt.figure(figsize=(10, 10))
        plt.imshow(image)

        show_mask(max_score_mask, plt.gca())

        plt.title(f"Highest score mask, Score: {max_score:.3f}", fontsize=18)
        plt.axis("off")
        plt.show()
    
    _log("Searching for inner bounding box...")

    bb = get_inner_bb(mask)

    if _plot:
        _log("Plotting inner mask for biggest score prediction...")

        plt.figure(figsize=(10, 10))
        plt.imshow(image)

        show_mask(max_score_mask, plt.gca())

        show_box(bb, plt.gca())

        plt.title(f"Inner Bounding Box (Area to crop)", fontsize=18)
        plt.axis("off")
        plt.show()

    # Crop image
    image_cropped = image[bb[1]:bb[3], bb[0]:bb[2]]

    if _plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(image_cropped)
        plt.title(f"Cropped Image", fontsize=18)
        plt.axis("off")
        plt.show()

    # Save image
    _log("Saving modified image...");
    
    new_filename = filename[:-4] + "_cropped.JPG"

    cv2.imwrite(new_filename, image_cropped);

    _metadata = pyexiv2.ImageMetadata(new_filename)
    _metadata.read()

    _log("Writing metadata...")
    try:
        metadata['Exif.Image.XResolution'] = _metadata['Exif.Image.XResolution']
        metadata['Exif.Image.YResolution'] = _metadata['Exif.Image.YResolution']
    except:
        # The tags didnt exist
        pass

    metadata['Exif.Photo.UserComment'] = pyexiv2.ExifTag("Exif.Photo.UserComment", "_quadrantrecon_marker")

    metadata.copy(_metadata)
    _metadata.write();


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Please use a path.")

        exit()
    
    if "--verbose" in sys.argv or "-v" in sys.argv:
        __DEBUG__ = True

    _log("Initializing QT.")
    QtWidgets.QApplication(sys.argv)

    main(sys.argv[1], True if "--plot" in sys.argv or "-p" in sys.argv else False)
