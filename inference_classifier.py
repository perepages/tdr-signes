import mediapipe as mp          # library that provides tools to create apps with a bit of AI
import cv2                      # library to initialize the camera
import pickle                   # library to save dataset, modules, info...
import numpy as np              # library that helps to do mathematical operations and convert veriables into others

model_dict = pickle.load(open('./FINAL/model.p', 'rb')) #loading the model
model = model_dict['model']


cap = cv2.VideoCapture(0) # iInitialize videocapture

# the objects for the landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.4) #module for detection

# Label for each letter (not all of them are detected due to some prolems)
labels_dict = {0: 'A', 1:'B', 2:'C', 3:"D", 4:"E", 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M',
                13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'}
while True:
    
    data_aux = []
    x_ = []
    y_ = []
    
    ret, frame = cap.read()
    
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
        
        for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)

        # looking for the corners of the rectangle to display
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10
        
        # looks for information and matches whats in the dict and whats in the screen
        prediction = model.predict([np.asarray(data_aux)[:42]])            
        predicted_character = labels_dict[int(prediction[0])]         
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4) #Creates he frame around the hand
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA) #displaying the letters

        
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()