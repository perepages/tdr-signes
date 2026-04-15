# ALL THE FILES THAT HAVE THE NUMBER (1) BEHIND THEIR NAME ARE RELATED TO
# THE CODE FOR SIGNS WITH THE TWO HANDS, THATS WHY SOMETIMES THE CODE VARIES
# AND SO DO THE COMMENTS.


import mediapipe as mp      # library that provides tools to create apps with a bit of AI
import cv2                  # library to initialize the camera
import pickle               # library to save dataset, modules, info...
import numpy as np          # library that helps to do mathematical operations and convert veriables into others
import time                 # library thet helps with time counting (creating delays in time)

# Load the model
model_dict = pickle.load(open('./FINAL/model.p', 'rb'))
model = model_dict['model']

# Initialize VideoCapture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.4)

# Define labels dictionary
labels_dict = {
    0: 'BON', 1: 'DIA', 2: 'TARDA', 3: "HOLA", 4: "ADEU", 
    5: 'QUIN ÉS EL TEU SENYAL?', 6: 'COM ET DIUS?', 7: 'MOLT DE GUST', 
    8: 'HO SENTO', 9: 'DEMA', 10: 'CATALUNYA', 11: 'GRACIES',
    12: 'NIT', 13: 'DE RES', 14: 'DISCULPA', 15: "D'ACORD", 16: "PRESENTAR", 
    17: 'FINS', 18: 'RECORDS', 19: 'A REVEURE'
}

last_predicted_character = None
last_predicted_time = None

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# Convert to RGB format because ony format accessible for mediapipe (just colors not really important)

    results = hands.process(frame_rgb) #detect all the landmarks into the image
    # loop for iteration into this landmarks. Look at every position
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, #image to draw
                hand_landmarks, #model output
                mp_hands.HAND_CONNECTIONS, #hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)
                data_aux.extend([landmark.x, landmark.y])

        if len(data_aux) == 42 or len(data_aux) == 84:  # Ensure we have the expected number of features
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            if len(data_aux) == 42:
                data_aux.extend([0] * 42)  # Pad with zeros if only one hand is detected

            prediction = model.predict([data_aux])
            predicted_label = int(prediction[0])

            if predicted_label in labels_dict:
                predicted_character = labels_dict[predicted_label]
            else:
                predicted_character = "Unknown"

            current_time = time.time()  # Establish time so later we can create some delays

            # Chech if last character is diferent than the previous
            if predicted_character != last_predicted_character:  
                last_predicted_character = predicted_character  # Update the character if so.
                last_predicted_time = current_time  # Establish time when the character is changed.

            # Check if there were 2 seconds of delay (need it).
            elif last_predicted_time and current_time - last_predicted_time >= 2:  
                # Print the text in the console.
                print(predicted_character)  
                last_predicted_time = None  # Restart time since last prediction


            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    else:
        last_predicted_character = None
        last_predicted_time = None

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
