import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings

# Suppress the specific deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*SymbolDatabase.GetPrototype.*")

# Directory containing the dataset
DATA_DIR = './dataset/train'

# Create a mapping of indices to labels
label_map = {idx: label for idx, label in enumerate(os.listdir(DATA_DIR))}

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Mediapipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Load the trained model
model_dict = pickle.load(open("model.p", "rb"))
model = model_dict["model"]

while True:
    data_aux = []
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Collect the hand landmark coordinates
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x)
            data_aux.append(y)

        # Make a prediction
        prediction = model.predict([np.asarray(data_aux)])
        predicted_char = label_map[prediction[0]]

        # Draw the fancy frame
        frame_height, frame_width, _ = frame.shape
        top_bar_height = 50
        bottom_bar_height = 100
        
        # Top bar (for webcam feed)
        cv2.rectangle(frame, (0, 0), (frame_width, top_bar_height), (0, 0, 0), -1)
        
        # Bottom bar (for predicted text)
        cv2.rectangle(frame, (0, frame_height - bottom_bar_height), (frame_width, frame_height), (0, 0, 0), -1)

        # Display the predicted character
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (255, 255, 255)
        cv2.putText(frame, f"Predicted char is: {predicted_char}", 
                    (50, frame_height - 40), font, font_scale, text_color, font_thickness)

    # Show the frame with the fancy borders and text
    cv2.imshow("Fancy Frame", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
