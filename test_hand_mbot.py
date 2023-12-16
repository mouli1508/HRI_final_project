import cv2
import mediapipe as mp
from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
import math


pi_ip = '192.168.1.112'  # Replace with the actual IP address of your Raspberry Pi

pi_factory = PiGPIOFactory(host=pi_ip)

#Create a servo object on pin 23
servo_arm = Servo(23, pin_factory=pi_factory)
servo_hand = Servo(24, pin_factory=pi_factory)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

def normalize_value(value, min_old, max_old, min_new, max_new):
    normalized_value = (value - min_old) / (max_old - min_old) * (max_new - min_new) + min_new
    return normalized_value

def euclidean_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detections
        results = hands.process(image)

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                          )

                # Get the landmarks of the middle finger
                middle_finger_base = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                index_finger_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_finger_tip = hand.landmark[mp_hands.HandLandmark.THUMB_TIP]

                # Convert landmarks to pixel coordinates
                h, w, _ = frame.shape
                base_point = (int(middle_finger_base.x * w), int(middle_finger_base.y * h))
                i_tip_point = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
                t_tip_point = (int(thumb_finger_tip.x * w), int(thumb_finger_tip.y * h))

                dist = euclidean_distance(i_tip_point, t_tip_point)

                n_dist = normalize_value(dist, 30, 250, -1, 1)
                n_base_point = normalize_value(base_point[1], 0, 480, -1, 1)

                if (n_base_point >= -1 and n_base_point <= 1 and n_dist >= -1 and n_dist <= 1):
                    servo_arm.value = round(n_base_point*-1, 1)
                    servo_hand.value = round(n_dist*-1, 1)
                    # time.sleep(1)

                # Draw a circle at the bottom of the middle finger
                cv2.circle(frame, base_point, 5, (255, 0, 0), -1)

        cv2.imshow('output', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            servo_hand.value=0
            servo_arm.value=0
            break

cap.release()
cv2.destroyAllWindows()
