import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

zoom_factor = 1.0
alpha_zoom = 0.1
alpha_face = 0.1
alpha_face_smooth = 0.4  
face_x, face_y = None, None
face_x_smooth, face_y_smooth = None, None   
move_threshold = 5

blur_enabled = False
blur_strength = 15

bar_x = 100
bar_width = 600
bar_height = 20
pallino_radius = 10
blur_min = 1
blur_max = 35

tracker_window_size = 500

def draw_blur_bar(frame, bar_x, bar_width, bar_height, pallino_radius, frame_width):
    bar_center_x = (frame_width - bar_width) // 2
    cv2.rectangle(frame, (bar_center_x, 50), (bar_center_x + bar_width, 50 + bar_height), (255, 255, 255), 2)
    cv2.circle(frame, (bar_x, 60), pallino_radius, (0, 0, 255), -1)

def mouse_callback(event, x, y, flags, param):
    global bar_x
    if event == cv2.EVENT_MOUSEMOVE:
        if 100 <= x <= 100 + bar_width and 50 <= y <= 70:
            bar_x = x
            global blur_strength
            blur_strength = int(np.interp(bar_x, [100, 100 + bar_width], [blur_min, blur_max]))

cv2.namedWindow("Tracker")
cv2.setMouseCallback("Tracker", mouse_callback)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            new_face_x = x + w // 2
            new_face_y = y + h // 2

            if face_x is None or face_y is None:
                face_x, face_y = new_face_x, new_face_y
            else:
                if abs(new_face_x - face_x) > move_threshold or abs(new_face_y - face_y) > move_threshold:
                    face_x = alpha_face * new_face_x + (1 - alpha_face) * face_x
                    face_y = alpha_face * new_face_y + (1 - alpha_face) * face_y
 
            if face_x_smooth is None or face_y_smooth is None:
                face_x_smooth, face_y_smooth = face_x, face_y
            else:
                face_x_smooth = alpha_face_smooth * face_x + (1 - alpha_face_smooth) * face_x_smooth
                face_y_smooth = alpha_face_smooth * face_y + (1 - alpha_face_smooth) * face_y_smooth

            
            radius = min(w, h) // 2   
            cv2.circle(frame, (int(face_x_smooth), int(face_y_smooth)), radius, (0, 0, 0), -1)  # Cerchio stabile sulla faccia

        distances = []
        num_hands = 0

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            for hand_landmarks in results.multi_hand_landmarks:
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
                distance_pixels = distance * max(frame.shape[0], frame.shape[1])
                distances.append(distance_pixels)

        if len(distances) > 0:
            target_zoom = 1 + (np.mean(distances) / 150)
            if num_hands == 2:
                target_zoom *= 1.3
            target_zoom = np.clip(target_zoom, 1.0, 3.0)
            zoom_factor = alpha_zoom * target_zoom + (1 - alpha_zoom) * zoom_factor

        height, width, _ = frame.shape
        min_side = min(width, height)
        crop_size = int(min_side / zoom_factor)

        center_x = int(face_x) if face_x is not None else width // 2
        center_y = int(face_y) if face_y is not None else height // 2

        x1 = max(center_x - crop_size // 2, 0)
        y1 = max(center_y - crop_size // 2, 0)
        x2 = min(center_x + crop_size // 2, width)
        y2 = min(center_y + crop_size // 2, height)

        if x2 - x1 < crop_size:
            x1 = max(x2 - crop_size, 0)
        if y2 - y1 < crop_size:
            y1 = max(y2 - crop_size, 0)

        zoomed_frame = frame[y1:y2, x1:x2]
        zoomed_frame = cv2.resize(zoomed_frame, (tracker_window_size, tracker_window_size), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Webcam", zoomed_frame)

        black_frame = np.zeros_like(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(black_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        blurred_frame = black_frame
        if blur_enabled:
            blurred_frame = cv2.GaussianBlur(black_frame, (blur_strength, blur_strength), 0)

      
        if face_x_smooth is not None and face_y_smooth is not None:
            size = 100  
            top_left = (int(face_x_smooth - size // 2), int(face_y_smooth - size // 2))
            bottom_right = (int(face_x_smooth + size // 2), int(face_y_smooth + size // 2))
            cv2.rectangle(black_frame, top_left, bottom_right, (255, 0, 0), 2) 

       
        cv2.imshow("Tracker", black_frame)

        draw_blur_bar(black_frame, bar_x, bar_width, bar_height, pallino_radius, 530)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('b'):
            blur_enabled = not blur_enabled

cap.release()
cv2.destroyAllWindows()
