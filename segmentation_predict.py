import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import cv2

from fit_ellipse import find_inscribed_ellipsiod

PROJECT_PATH = Path()
model_name = 'yolov8n-seg.pt'
folder_path = PROJECT_PATH.joinpath("data", "labeled_data")
results_path = PROJECT_PATH.joinpath("data", "results")
images = [name for name in folder_path.glob("**/*.jpg")]

# Load a model
model = YOLO("segmentation_model.pt")
# Inference
inference = model(images, device="cpu")

inference_results = {"image": [], "is_detected": [], "a": [], "b": []}
for img_path, result in zip(images, inference):
    print(f"Processing {img_path.stem}...")
    if result.masks is None:
        inference_results["image"].append(str(img_path))
        inference_results["is_detected"].append(False)
        inference_results["a"].append(None)
        inference_results["b"].append(None)
        break
    masks = result.masks.xy
    for idx, mask in enumerate(masks):
        dic_results = find_inscribed_ellipsiod(mask)
        # Write down results
        inference_results["image"].append(str(img_path))
        inference_results["is_detected"].append(True)
        inference_results["a"].append(dic_results["a"])
        inference_results["b"].append(dic_results["b"])

        img = cv2.imread(str(img_path))
        for coord in dic_results["ellipse"]:
            img = cv2.circle(img, (int(coord[0]), int(coord[1])), 1, (0, 0, 255), -1)

        img = cv2.polylines(img, [dic_results["hull"]], True, (0, 0, 255), 1)
        cv2.imwrite(str(results_path.joinpath(img_path.name)), img)
        # cv2.imshow("Image", img)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
        #
inference_results = pd.DataFrame(inference_results)
inference_results.to_excel("inference_results.xlsx", index=False)

