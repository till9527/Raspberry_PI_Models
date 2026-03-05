import numpy as np
import cv2
import socket
from flask import Flask, Response
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.post_processors import pp_od_yolo_ultralytics

# --- 1. SET UP THE WEB SERVER ---
app = Flask(__name__)

# --- 2. YOUR AI MODEL ---
class YOLO(Model):
    def __init__(self):
        super().__init__(
            model_file="best_imx_model_yolo8v2/packerOut.zip",
            model_type=MODEL_TYPE.CONVERTED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(
            "best_imx_model/labels.txt",
            dtype=str,
            delimiter="\n",
        )

    def post_process(self, output_tensors):
        return pp_od_yolo_ultralytics(output_tensors)

# --- 3. THE VIDEO STREAM GENERATOR ---
def generate_frames():
    device = AiCamera(image_size=(640, 480), frame_rate=16)
    model = YOLO()
    device.deploy(model)
    annotator = Annotator()

    LOWER_BOUND = np.array([10, 0, 0])
    UPPER_BOUND = np.array([45, 255, 255])

    with device as stream:
        for frame in stream:
            clean_img = frame.image.copy()

            # AI Annotation
            detections = frame.detections[frame.detections.confidence > 0.0]
            labels = [f"{model.labels[class_id]}: {score:0.2f}" for _, score, class_id, _ in detections]
            annotator.annotate_boxes(frame, detections, labels=labels, alpha=0.3, corner_radius=10)
            
            # Thresholding
            roi = clean_img[320:480, 0:640] 
            hsv_img = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            binary_view = cv2.inRange(hsv_img, LOWER_BOUND, UPPER_BOUND)
            
            # Convert binary image (1 channel) to BGR (3 channels) so we can stack it
            binary_3ch = cv2.cvtColor(binary_view, cv2.COLOR_GRAY2BGR)
            
            # Stack images vertically (Main Feed on top, Threshold on bottom)
            combined_view = cv2.vconcat([frame.image, binary_3ch])

            # Compress to JPEG
            ret, buffer = cv2.imencode('.jpg', combined_view)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            # Yield the frame to the web server
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- 4. THE WEB ROUTE ---
@app.route('/')
def video_feed():
    # This route tells the browser to expect a continuous stream of JPEGs
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 5. GET IP AND RUN ---
def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

if __name__ == '__main__':
    pi_ip = get_ip()
    print("\n" + "="*50)
    print("🚀 AI STREAM IS LIVE!")
    print(f"👉 Click here or paste this into your browser: http://{pi_ip}:5000")
    print("="*50 + "\n")
    
    # Run the server on port 5000
    app.run(host='0.0.0.0', port=5000, threaded=True)