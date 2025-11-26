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
model = YOLO("best_ncnn_model_64")  # <<< 3. Use the new, robust path

# Create a VideoCapture object for the default webcam (index 0)
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,120)
# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # If the frame was not read successfully, break the loop
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # --- YOLOv8 Inference ---
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()
    # --------------------------
    #annotated_frame=cv2.resize(annotated_frame,(160,120))

    # Display the annotated frame
    cv2.imshow("Webcam Feed", annotated_frame)

    # Wait for 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release and destroy
cap.release()
cv2.destroyAllWindows()
