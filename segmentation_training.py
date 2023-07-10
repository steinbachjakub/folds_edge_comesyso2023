from ultralytics import YOLO
from pathlib import Path

PROJECT_PATH = Path()
model_name = 'yolov8n-seg.pt'
folder_name = "test"
yaml_path = Path("data", "labeled_data", "data.yaml").absolute()
epochs = 100

if __name__ == "__main__":
    # Load a model
    model = YOLO(model_name)  # load a pretrained model (recommended for training)
    # Train the model
    model.train(data=yaml_path, epochs=epochs, device=0, name=f"{folder_name}_{model_name.split('.')[0]}_{epochs}")

