# In this file we detect the landmarks and classify them with arrays that contain all the info
import os                       # library to create files and directories
import pickle                   # library to save dataset, modules, info...
import mediapipe as mp          # library that provides tools to create apps with a bit of AI
import cv2                      # library to initialize the camera


# Defining the objects for the landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) #module for detection


DATA_DIR = './FINAL/data'

# Variables that store the info
data = [] #info
labels = [] #category (letter / sign)


# We take all the info from the images 
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)# Convert to RGB format because ony format accessible for mediapipe (just colors not really important)
        
        results = hands.process(img_rgb) #detect all the landmarks into the image
        # loop for iteration into this landmarks. Look at every position
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
            
            # Create the dataset file to store all this info
            data.append(data_aux)
            labels.append(dir_)
            
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        