import numpy as np
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.post_processors import pp_od_yolo_ultralytics
from modlib.models.zoo import YOLO11n
import cv2

class YOLO(Model):
    """YOLO model for IMX500 deployment."""

    def __init__(self):
        """Initialize the YOLO model for IMX500 deployment."""
        super().__init__(
            model_file="best_imx_model_yolo8v2/packerOut.zip",  # replace with proper directory
            model_type=MODEL_TYPE.CONVERTED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )

        self.labels = np.genfromtxt(
            "best_imx_model/labels.txt",  # replace with proper directory
            dtype=str,
            delimiter="\n",
        )

    def post_process(self, output_tensors):
        """Post-process the output tensors for object detection."""
        return pp_od_yolo_ultralytics(output_tensors)


device = AiCamera(image_size=(640,480),frame_rate=16)  # Optimal frame rate for maximum DPS of the YOLO model running on the AI Camera
model = YOLO()
device.deploy(model)

annotator = Annotator()

with device as stream:
    for frame in stream:
        detections = frame.detections[frame.detections.confidence > 0.0]
        labels = [f"{model.labels[class_id]}: {score:0.2f}" for _, score, class_id, _ in detections]

        annotator.annotate_boxes(frame, detections, labels=labels, alpha=0.3, corner_radius=10)
        display_img = cv2.resize(frame.image,(640,480))
        cv2.imshow("Result",display_img)
        cv2.waitKey(1)
