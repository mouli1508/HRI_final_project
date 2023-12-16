import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

import mediapipe as mp

from supervision.draw.color import ColorPalette
# from supervision.tools.detections import Detections, BoxAnnotator

import supervision as sv


class ObjectDetection:

    def __init__(self, capture_index):

        self.capture_index = capture_index

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

    def mid_points(self, results, frame):

        for result in results:
            # val = result.boxes.xyxy.cpu().numpy()
            for val in result.boxes.xyxy.cpu().numpy():
                try:
                    x1 = val[0]
                    y1 = val[1]
                    x2 = val[2]
                    y2 = val[3]

                    x_mid = (x1 + x2) / 2
                    y_mid = (y1 + y2) / 2

                    cv2.circle(frame, (int(x_mid), int(y_mid)), 4, (255, 0, 0), -1)
                except:
                    pass

    def __call__(self):

        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # width and height of USB camera = 640 x 480
        with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
            while cap.isOpened():

                start_time = time()

                ret, frame = cap.read()
                assert ret

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                hand_results = hands.process(rgb_frame)

                if hand_results.multi_hand_landmarks:
                    for num, hand in enumerate(hand_results.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2,
                                                                         circle_radius=4),
                                                  mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2,
                                                                         circle_radius=2),
                                                  )

                results = self.predict(frame)
                frame = self.plot_bboxes(results, frame)

                end_time = time()
                fps = 1 / np.round(end_time - start_time, 2)

                cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

                cv2.imshow('YOLOv8 Detection', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


detector = ObjectDetection(capture_index=1)
detector()