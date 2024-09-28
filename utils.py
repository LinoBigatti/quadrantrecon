import numpy as np

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
