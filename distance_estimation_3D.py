from realsense_camera import *

import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

from supervision.draw.color import ColorPalette
# from supervision.tools.detections import Detections, BoxAnnotator

import supervision as sv


class ObjectDetection:

    def __init__(self):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.model = self.load_model()

        self.CLASS_NAMES_DICT = self.model.model.names

        self.box_annotator = sv.BoxAnnotator(color=ColorPalette.default(), thickness=3, text_thickness=3,
                                             text_scale=1.5)

    def load_model(self):

        model = YOLO("yolov8m.pt")  # load a pretrained YOLOv8n model
        model.fuse()

        return model

    def predict(self, frame):

        results = self.model(frame)

        return results

    def plot_bboxes(self, results, frame):

        xyxys = []
        confidences = []
        class_ids = []

        # Extract detections for person class
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)

            if class_id == 0:
                xyxys.append(result.boxes.xyxy.cpu().numpy())
                confidences.append(result.boxes.conf.cpu().numpy())
                class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

        # Setup detections for visualization
        detections = sv.Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int),
        )

        # Format custom labels
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                       for _, mask, confidence, class_id, tracker_id
                       in detections]

        # Annotate and display frame
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)

        return frame

    def mid_points(self, results, frame, depth_frame):
        for result in results:
            for val in result.boxes.xyxy.cpu().numpy():
                try:
                    x1 = val[0]
                    y1 = val[1]
                    x2 = val[2]
                    y2 = val[3]

                    x_mid = (x1 + x2) // 2
                    y_mid = (y1 + y2) // 2

                    # Ensure the midpoint coordinates are within the depth frame dimensions
                    x_mid = max(0, min(x_mid, depth_frame.shape[1] - 1))
                    y_mid = max(0, min(y_mid, depth_frame.shape[0] - 1))

                    # Convert to integer indices
                    x_mid = int(x_mid)
                    y_mid = int(y_mid)

                    depth_mm = depth_frame[y_mid, x_mid]

                    cv2.circle(frame, (int(x_mid), int(y_mid)), 4, (255, 0, 0), -1)

                    cv2.putText(frame, "{} cm".format(depth_mm / 10), (x_mid + 5, y_mid + 60), 0, 1, (0, 255, 0),
                                2)

                except Exception as e:
                    print(f"Error in mid_points: {e}")
                    pass

    def __call__(self):

        rs = RealsenseCamera()
        assert rs

        # width and height of USB camera = 640 x 480

        while True:

            start_time = time()

            _, frame, depth_frame = rs.get_frame_stream()

            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)

            self.mid_points(results, frame, depth_frame)

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('YOLOv8 Detection', frame)
            # cv2.imshow('depth frame', depth_colormap)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        rs.release()
        cv2.destroyAllWindows()


detector = ObjectDetection()
detector()
