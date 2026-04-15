from customtkinter import * # library for graphic interface tkinter
import cv2                  # library to initialize the camera
import mediapipe as mp      # library that provides tools to create apps with a bit of AI
import pickle               # library that provides modules to help store data
import numpy as np          # library that helps to do mathematical operations and convert veriables into others
from PIL import Image       # library linked with tkinter to insert images/videos to interfaces
import time                 # library thet helps with time counting (creating delays in time)
import pyttsx3              # library that provides audios (we haven't use it but we explain it eitherway)

# This commented parts correspond to a part of the code used to convert
# the text detected into audio. However, we do not used because it is
# an extra feature that makes our code crash, although we leave it here so
# people that look at the code know that it is possible and that we tried :)

#class TextToSpeech:
#    engine: pyttsx3.Engine
#    def __init__(self, voice, rate: int, volume: float):
#        self.engine = pyttsx3.init()
#        if voice:
#            self.engine.setProperty('voice', voice)
#        self.engine.setProperty('rate', rate)
#        self.engine.setProperty('volume', volume)
#    def list_available_voices(self):
#        voices: list = self.engine.getProperty('voices')
#        for i, voice in enumerate(voices):
#            languages = voice.languages[0] if voice.languages else 'unknown'
#            print(f'{i+1}. {voice.name}, {voice.age}: {languages} ({voice.gender}) [{voice.id}]')
#    def text_to_speech(self, text:str, save:bool=False, file_name='output.mp3'):
#        self.engine.say(text)
#        print('Speaking...')
#        if save:
#            self.engine.save_to_file(text, file_name)
#        self.engine.runAndWait()
#            # HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-ES_HELENA_11.0


# Load both models (letter and sign detection)
letter_model_dict = pickle.load(open('./FINAL/model.p', 'rb'))
letter_model = letter_model_dict['model']

sign_model_dict = pickle.load(open('./FINAL/model.p1', 'rb'))
sign_model = sign_model_dict['model']

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.4)

# Dictionaries for letter and sign labels
letter_labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: "D", 4: "E", 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
                10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
                19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

sign_labels_dict = {
    0: 'BON', 1: 'DIA', 2: 'TARDA', 3: "HOLA", 4: "ADEU", 
    5: 'QUIN ÉS EL TEU SENYAL?', 6: 'COM ET DIUS?', 7: 'MOLT DE GUST', 
    8: 'HO SENTO', 9: 'DEMA', 10: 'CATALUNYA', 11: 'GRACIES',
    12: 'NIT', 13: 'DE RES', 14: 'DISCULPA', 15: "D'ACORD", 
    16: "PRESENTAR", 17: 'FINS', 18: 'RECORDS', 19: 'A REVEURE'
}

# Variables to control predictions (characters and models to change)
last_predicted_character = None
last_predicted_time = None
current_model = None
current_labels = None
detected_sequence = ""  # Stores the accumulated sequence of detected letters or signs

# Tkinter app setup
app = CTk()
app.geometry("1000x600")
app.title("TDR Signes")
set_default_color_theme("green")

# Main frame
main_frame = CTkFrame(app, fg_color="#302c2c")
main_frame.pack(fill="both", expand=True, padx=20, pady=20)

# Camera frame
camera_frame = CTkFrame(main_frame)
camera_frame.pack(side=LEFT, padx=10, pady=10, expand=True)

# Label to display the camera
label = CTkLabel(camera_frame, text="")
label.pack(anchor="center", expand=True)

# Button frame
button_frame = CTkFrame(main_frame, fg_color="#302c2c")
button_frame.pack(side=RIGHT, padx=20, pady=10)

# Button actions
def switch_to_sign_model():
    global current_model, current_labels
    current_model = sign_model
    current_labels = sign_labels_dict
    prediction_label.configure(text="Canviat al model de signes")

def switch_to_letter_model():
    global current_model, current_labels
    current_model = letter_model
    current_labels = letter_labels_dict
    prediction_label.configure(text="Canviat al model de lletres")

def no_detection():
    global current_model, current_labels
    current_model = None
    current_labels = None
    prediction_label.configure(text="")
    sequence_label.configure(text="")

# Buttons with positions and commands
button1 = CTkButton(button_frame, text="SIGNES", command=switch_to_sign_model)
button1.pack(pady=10)

button2 = CTkButton(button_frame, text="LLETRES", command=switch_to_letter_model)
button2.pack(pady=10)

button3 = CTkButton(button_frame, text="REINICIAR", command=no_detection)
button3.pack(pady=10)

# Prediction label to display the sequence
sequence_label = CTkLabel(app, text="", font=("Arial", 20), wraplength=900)
sequence_label.pack(side=BOTTOM, pady=10)

# Label for the last prediction
prediction_label = CTkLabel(app, text="", font=("Arial", 24))
prediction_label.pack(side=BOTTOM, pady=10)

# Camera initializing setup
def cam():
    wCam, hCam = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    def update_frame(): # Keep updating the frame so it shows tha real life video
        global last_predicted_character, last_predicted_time, current_model, current_labels, detected_sequence

        success, frame = cap.read()
        if not success:
            print("Error: Cannot access camera.")
            return

        data_aux = []
        x_ = []
        y_ = []
        frame = cv2.flip(frame, 1)  # Invert the image to it looks good
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks: #conditions for functions when detecting the positions of the hands
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)
                    data_aux.extend([landmark.x, landmark.y])

            # Ensure the input has the correct number of features for the model
            # For a single hand we need 42 inputs (positions of the joints of the fingers)
            # For the two hands we need 84 inputs
            # If we insert more inputs than we need, the program crashes
            if current_model == letter_model and len(data_aux) > 42:
                data_aux = data_aux[:42]  # Trim to 42 features for letters
            elif current_model == sign_model and len(data_aux) == 42:
                data_aux.extend([0] * 42)  # Pad to 84 features for signs

            # Predict using the current model
            if current_model and len(data_aux) == 42 or len(data_aux) == 84:
                prediction = current_model.predict([np.asarray(data_aux)])
                predicted_label = int(prediction[0])

                # If there is something that the program doesn't get = "Unknown"
                predicted_character = current_labels.get(predicted_label, "Unknown")

                current_time = time.time()

                # Through the following conditions we diferenciate the characters so we display
                # on the screen no more tha one same character at a time
                if predicted_character != last_predicted_character:
                    last_predicted_character = predicted_character
                    last_predicted_time = current_time

                    # Append the new character to the sequence
                    detected_sequence += predicted_character + " "
                    length_ds = len(detected_sequence)
                    if length_ds>30: #Define a limit of characters to restart and not have an infinite display of characters
                        detected_sequence= " "
                    sequence_label.configure(text=detected_sequence)  # Update the sequence label with new labels
                    prediction_label.configure(text=predicted_character)  # Update the latest prediction label


                # Commented code regarding the audio converting code

                #if __name__ == '__main__':
                #    tts = TextToSpeech('HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-ES_HELENA_11.0', 200, 1.0)
                #    tts.list_available_voices()
                #    tts.text_to_speech(predicted_character)
                
                
                # Show the prediction in the label screen
                elif last_predicted_time and current_time - last_predicted_time >= 8:
                    prediction_label.configure(text=predicted_character)  
                    last_predicted_time = None

                # Draw rectangle around hand and display prediction
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        # Convert frame for display in the Tkinter UI
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = CTkImage(light_image=img_pil, size=(wCam, hCam))

        label.imgtk = imgtk
        label.configure(image=imgtk)
        label.after(10, update_frame)

    update_frame()

cam()  # Start the camera immediately

app.mainloop()
