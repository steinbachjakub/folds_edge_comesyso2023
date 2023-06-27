"""
Script with the detection algorithm. It takes the trained settings saved in .//trained_model//trained_model.pt and uses
it to detect the vocal folds region in the dataset images. The bounding boxes are then saved in csv file.
"""

# Imports
import torch
import pandas as pd
import cv2
from pathlib import Path


# Global Variables
PATH_RAW_IMAGES = Path("data", "raw", "images")
PATH_CROPPED_IMAGES = Path("data", "cropped_images")
PATH_CROPPED_IMAGES.mkdir(exist_ok=True, parents=True)
# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='trained_model/trained_model.pt')
# Images
imgs = list(PATH_RAW_IMAGES.glob("*.jpg"))  # batch of images
# Inference and saving the coordinates of the bounding box
for index, img in enumerate(imgs[:]):
    print(f"Processing {index}/{len(imgs)}...")
    result = model(img)
    df_result = result.pandas().xyxy[0]
    # Cropping the images based on the bounding boxes
    for i in range(df_result.shape[0]):
        # Saving the position of the bounding box
        yolo_file_name = img.stem
        yolo_xmin = int(df_result.xmin.values[i])
        yolo_xmax = int(df_result.xmax.values[i])
        yolo_ymin = int(df_result.ymin.values[i])
        yolo_ymax = int(df_result.ymax.values[i])

        # Uploading the image and cropping it along the bounding box
        raw_image = cv2.imread(str(img))
        cropped_image = raw_image[yolo_ymin:yolo_ymax + 1, yolo_xmin:yolo_xmax + 1]

        # # Display both images
        # cv2.rectangle(raw_image, (yolo_xmin, yolo_ymin), (yolo_xmax, yolo_ymax), (0, 255, 255), 1)
        # cv2.imshow("original", raw_image)
        # cv2.imshow("cropped", cropped_image)
        # cv2.waitKey(0)

        # Save the cropped image
        cv2.imwrite(str(PATH_CROPPED_IMAGES.joinpath(f"{img.stem}_{i:04}.jpg")), cropped_image)
        cv2.destroyAllWindows()

