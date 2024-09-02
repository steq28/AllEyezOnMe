import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

DATA_DIR = './dataset/train'

data = []
labels = []

# Step 1: Create a mapping from label strings to integers
label_map = {label: idx for idx, label in enumerate(os.listdir(DATA_DIR))}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

for _dir in os.listdir(DATA_DIR):
    print("Processing:", _dir)
    for file in os.listdir(os.path.join(DATA_DIR, _dir)):
        data_aux = []

        img = cv2.imread(DATA_DIR+"/"+_dir+"/"+file)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            # Only use the first hand detected
            hand_landmarks = results.multi_hand_landmarks[0]
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)

            data.append(data_aux)
            labels.append(label_map[_dir])

# Save the data
with open("data.pickle", "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

print("Label Mapping:", label_map)
