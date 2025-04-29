import torch
from pathlib import Path

# Configuration
DATA_YAML = "C:\\Users\\LENOVO\\Desktop\\cvvv\\boycott_dataset\\boycott.yaml"
MODEL_TYPE = "yolov5s"  # Options: yolov5s, yolov5m, yolov5l, yolov5x
EPOCHS = 50
BATCH_SIZE = 8  # Adjust based on your system's capability
IMG_SIZE = 640
DEVICE = "cpu"  # Change to "cuda" if you have a GPU

def main():
    # Verify YAML exists
    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(f"YAML file not found at {DATA_YAML}")

    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', MODEL_TYPE, pretrained=True)

    # Train the model
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        workers=0,  # Important for CPU
        project="runs/train",
        name="boycott_logo_detection",
        exist_ok=True
    )

    print("Training completed successfully!")
    print(f"Results saved to: runs/train/boycott_logo_detection")

if __name__ == "__main__":
    main()
