"""
Script with the detection algorithm. It takes the trained settings saved in .//trained_model//trained_model.pt and uses
it to detect the vocal folds region in the dataset images. The bounding boxes are then saved in csv file.
"""

# Imports
import torch
from pathlib import Path


# Global Variables
PATH_IMAGES = Path("data", "images")
# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='trained_model/trained_model.pt')
# Images
imgs = [PATH_IMAGES.joinpath("0a9a0c45-3_poop_stitna_zlaza_frame_582.jpg")]  # batch of images

# Inference
results = model(imgs)
print(results.show())
