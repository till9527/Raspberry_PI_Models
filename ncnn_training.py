import cv2
from ultralytics import YOLO
from pathlib import Path  # <<< 1. Import Path

# --- Build Absolute Path to Model ---
# This gets the directory where your .py script is located
SCRIPT_DIR = Path(__file__).resolve().parent
# This joins that directory path with your model path
MODEL_PATH = SCRIPT_DIR / "model" / "best.pt"  # <<< 2. Create the path
# ------------------------------------


# Load your custom YOLOv8 model using the absolute path
model = YOLO(MODEL_PATH)  # <<< 3. Use the new, robust path
model.export(format="ncnn",imgsz=96)