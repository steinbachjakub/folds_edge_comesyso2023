"""
Script that takes cropped images with folds as an input and try to find the "edges" of the folds to estimate
the fold openess
"""

# Imports
import cv2
import numpy as np
from pathlib import Path
import os


def auto_canny_edge_detection(image, sigma=0.33):
    md = np.median(image)
    lower_value = int(max(0, (1.0-sigma) * md))
    upper_value = int(min(255, (1.0+sigma) * md))
    return cv2.Canny(image, lower_value, upper_value)


# Global variables
PATH_CROPPED_IMAGES = Path("data", "cropped_images")
PATH_DETECTION = Path("data", "edge_detection")
RESIZE = (100, 200)

# Creating subfolders if needed
for path in [PATH_CROPPED_IMAGES, PATH_DETECTION]:
    path.mkdir(exist_ok=True, parents=True)

# List of all cropped images
all_cropped_images = list(PATH_CROPPED_IMAGES.glob("*.jpg"))
# all_cropped_images = [PATH_CROPPED_IMAGES.joinpath("3d41d99b-2_pareza_frame_355_0000.jpg"), ]

for index, img in enumerate(all_cropped_images):
    image = cv2.imread(str(img))
    resized = cv2.resize(image, RESIZE)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(equ, (7, 7), 0)
    perc = np.percentile(blurred, 20)
    _, bw = cv2.threshold(blurred, perc, 255, cv2.THRESH_BINARY_INV)
    edge = auto_canny_edge_detection(blurred, sigma=0.33)

    stack = np.hstack((gray, equ, blurred, bw, edge))  # stacking images side-by-side
    if not cv2.imwrite(str(PATH_DETECTION.joinpath(img.name)), stack):
        raise Exception("Could not write image")

# Change the current directory
# to specified directory
directory = Path("data")
os.chdir(directory)

if not cv2.imwrite('modifications.jpg', stack):
    raise Exception("Could not write image")
cv2.destroyAllWindows()
