# 🤟 Real-Time Sign Language AI Translator

A real-time Computer Vision desktop application that translates Catalan Sign Language (LSC) and the alphabet into text using a standard webcam. 


## ✨ Features
* **Real-Time Translation:** Detects hand signs and displays the corresponding text instantly.
* **Custom Machine Learning Pipeline:** Built from scratch—from image data collection and feature extraction to model training and deployment.
* **Dual Hand Support:** Capable of detecting both single-hand alphabet characters and complex two-hand signs.
* **Modern GUI:** Features an interactive and sleek user interface built with CustomTkinter.

## 🛠️ Tech Stack
* **Language:** Python
* **Computer Vision:** OpenCV, Google MediaPipe (for spatial Hand Landmark detection)
* **Machine Learning:** Scikit-Learn (Random Forest Classifier)
* **UI Framework:** CustomTkinter, Pillow

## 🧠 How It Works
1. **Feature Extraction:** MediaPipe extracts up to 84 spatial coordinates (x, y) from the joints of the hands in the video feed.
2. **Classification:** A trained Random Forest model analyzes these coordinates to predict the exact sign.
3. **Display:** The prediction is smoothed and displayed in the UI.

## 💻 How to Run Locally

1. Clone the repository:
```bash
git clone https://github.com/[YourUsername]/[RepoName].git
