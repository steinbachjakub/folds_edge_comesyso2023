from ultralytics import YOLO
from pathlib import Path
import cv2

PROJECT_PATH = Path()
model_name = 'yolov8n-seg.pt'
folder_path = PROJECT_PATH.joinpath("data", "labeled_data")
images = [str(name) for name in folder_path.glob("**/*.jpg")]

# Load a model
model = YOLO("best.pt")  # load a pretrained model (recommended for training)
# model.load(weights="best.pt")
# Inference
results = model(images[:5], device="cpu")

for img_path, result in zip(images, results):
    print(img_path)
    img = cv2.imread(img_path)
    masks = result.masks.xy
    for mask in masks:
        for coord in mask:
            img = cv2.circle(img, (int(coord[0]), int(coord[1])), 1, (0, 0, 255), -1)
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
print(results[0].masks.xy)
# np.savetxt("tensor.txt", results[0].masks.data)

