import cv2
import mediapipe as mp
import math

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

                thumb_finger_ip = hand.landmark[mp_hands.HandLandmark.THUMB_IP]
                index_finger_dip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
                middle_finger_dip = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
                middle_finger_tip = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_finger_dip = hand.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
                ring_finger_tip = hand.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

                # Convert landmarks to pixel coordinates
                h, w, _ = frame.shape
                m_tip_point = (int(middle_finger_tip.x * w), int(middle_finger_tip.y * h))
                m_base_point = (int(middle_finger_base.x * w), int(middle_finger_base.y * h))
                m_dip_point = (int(middle_finger_dip.x * w), int(middle_finger_dip.y * h))

                i_tip_point = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
                i_dip_point = (int(index_finger_dip.x * w), int(index_finger_dip.y * h))

                t_tip_point = (int(thumb_finger_tip.x * w), int(thumb_finger_tip.y * h))
                t_dip_point = (int(thumb_finger_ip.x * w), int(thumb_finger_ip.y * h))

                r_tip_point = (int(ring_finger_tip.x * w), int(ring_finger_tip.y) * h)
                r_dip_point = (int(ring_finger_dip.x * w), int(ring_finger_dip.y) * h)

                dist_thumb = euclidean_distance(t_tip_point, t_dip_point)
                dist_index = euclidean_distance(i_tip_point, i_dip_point)
                dist_ring = euclidean_distance(r_tip_point, r_dip_point)
                dist_middle = euclidean_distance(m_tip_point, m_dip_point)

                # print(dist_thumb, dist_index, dist_middle, dist_ring)
                # 38 20 20 13

                if (dist_thumb < 38 and dist_index <20 and dist_ring <20 and dist_middle<13):
                    print("GRAB!")

                dist = euclidean_distance(i_tip_point, t_tip_point)
                # print(dist)

                n_dist = normalize_value(dist, 30, 250, -1, 1)
                n_base_point = normalize_value(m_base_point[1], 0, 480, -1, 1)

                # Draw a circle at the bottom of the middle finger
                cv2.circle(frame, m_base_point, 5, (255, 0, 0), -1)

        cv2.imshow('output', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
