import torch
from pathlib import Path

# Configuration
TEST_IMAGES_DIR = "C:/Users/LENOVO/Desktop/cvvv/image_test/"  # Your test images
MODEL_PATH = "C:/Users/LENOVO/Desktop/cvvv/runs/train/boycott_logo_detection/weights/best.pt"
OUTPUT_DIR = "C:/Users/LENOVO/Desktop/cvvv/runs/detect/"     # Detection results
CONF_THRESH = 0.5  # Confidence threshold (0-1)
IOU_THRESH = 0.45  # Intersection Over Union threshold

def main():
    # Verify paths exist
    if not Path(TEST_IMAGES_DIR).exists():
        raise FileNotFoundError(f"Test images directory not found at {TEST_IMAGES_DIR}")
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    # Load custom trained model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
    
    # Set model parameters
    model.conf = CONF_THRESH  # Confidence threshold
    model.iou = IOU_THRESH    # NMS IoU threshold
    
    # Perform detection
    results = model(TEST_IMAGES_DIR)
    
    # Save results
    results.save(save_dir=OUTPUT_DIR)
    print(f"Detection results saved to {OUTPUT_DIR}")
    
    # Show results (optional)
    results.show()

if __name__ == "__main__":
    main()