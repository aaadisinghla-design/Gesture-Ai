import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Filter state variables
filters = ['none', 'grayscale', 'sepia', 'negative', 'blur']
current_filter_idx = 0
cooldown_frames = 15  # Prevent rapid accidental switching
cooldown_counter = 0

# Image capture flag
photo_taken_counter = 0

def apply_filter(frame, filter_name):
    if filter_name == 'grayscale':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif filter_name == 'sepia':
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sepia = cv2.transform(frame, kernel)
        return np.clip(sepia, 0, 255).astype(np.uint8)
    elif filter_name == 'negative':
        return cv2.bitwise_not(frame)
    elif filter_name == 'blur':
        return cv2.GaussianBlur(frame, (15, 15), 0)
    return frame

def get_distance(pt1, pt2):
    return np.sqrt((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2)

# Open webcam
cap = cv2.VideoCapture(0)

print("Starting Gesture-Controlled Photo App...")
print("- Thumb + Index: Capture Photo")
print("- Thumb + Middle / Ring / Pinky: Cycle Filters")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for a mirror view
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Process hand landmarks
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture_detected = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract key landmarks
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Threshold for considering fingers "touching"
            threshold = 0.04

            # Check gesture logic
            if get_distance(thumb_tip, index_tip) < threshold:
                gesture_detected = 'capture'
            elif (get_distance(thumb_tip, middle_tip) < threshold or 
                  get_distance(thumb_tip, ring_tip) < threshold or 
                  get_distance(thumb_tip, pinky_tip) < threshold):
                gesture_detected = 'filter'

    # Apply the current active filter to the display stream
    filtered_frame = apply_filter(frame.copy(), filters[current_filter_idx])

    # Handle action states and cooldowns
    if cooldown_counter > 0:
        cooldown_counter -= 1

    if cooldown_counter == 0 and gesture_detected is not None:
        if gesture_detected == 'capture':
            filename = f"photo_{photo_taken_counter}.png"
            cv2.imwrite(filename, filtered_frame)
            print(f"[SUCCESS] Saved snapshot: {filename}")
            photo_taken_counter += 1
            cooldown_counter = 30  # Longer cooldown for photo capture visual feedback
            
            # Flash effect on screen
            filtered_frame[:, :, :] = 255
            
        elif gesture_detected == 'filter':
            current_filter_idx = (current_filter_idx + 1) % len(filters)
            print(f"[FILTERCHANGED] Switched to: {filters[current_filter_idx]}")
            cooldown_counter = cooldown_frames

    # Render UI Overlay info
    cv2.putText(filtered_frame, f"Active Filter: {filters[current_filter_idx].upper()}", 
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Gesture Photo Hub", filtered_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
