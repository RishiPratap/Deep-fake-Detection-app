import streamlit as st
import cv2
import numpy as np
from tempfile import NamedTemporaryFile
from tensorflow.lite.python.interpreter import Interpreter
from PIL import Image
from datetime import datetime
import os

# Load the TFLite model
interpreter = Interpreter(model_path="deepfake_detection_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Function to preprocess video frames
def preprocess_frames(frames):
    preprocessed_frames = []
    for frame in frames:
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preprocessed_frames.append(frame)
    return np.array(preprocessed_frames)

# Function to extract frames from video
def extract_frames(video_bytes, num_frames=10):
    frames = []
    with NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_bytes)
        tmp_file_path = tmp_file.name
    
    cap = cv2.VideoCapture(tmp_file_path)
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    # Delete temporary file
    os.unlink(tmp_file_path)
    
    return frames

# Function to predict if the video is real or deepfake
def predict_video(uploaded_file, status_log):
    video_bytes = uploaded_file.read()  # Read byte data from file object
    frames = extract_frames(video_bytes, num_frames=10)
    preprocessed_frames = preprocess_frames(frames)
    preprocessed_frames = preprocessed_frames.reshape((1, 10, 224, 224, 3))

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], preprocessed_frames.astype('float32'))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    accuracy = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    result = "Deepfake" if prediction[0][0] > 0.5 else "Real"

    status_log.text(f"[{timestamp}] - The video is {result} with {accuracy * 100:.2f}% accuracy.\n")

    return result, accuracy

def main():
    st.title("Deepfake Detection App")

    # Section 1: Title and Navigation Bar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ("Home", "About"))

    # Section 2: Video and Model Detection
    if page == "Home":
        st.header("Video Player and Model Detection")
        uploaded_file = st.file_uploader("Upload a video file:", type=["mp4"])

        # Logs Section
        st.header("Logs:")
        status_log = st.empty()

        if uploaded_file is not None:
            status_log.text(f"[{timestamp}] - Video uploaded successfully.\n")
            st.video(uploaded_file)

            if st.button("Detect"):
                status_log.text(f"[{timestamp}] - Detecting video...\n")
                result, accuracy = predict_video(uploaded_file, status_log)
                
                # Table displaying model name, live accuracy, and status
                st.subheader("Model Detection Status")
                data = {"Model Name": ["Resnet+LSTM"], "Accuracy": [f"{accuracy * 100:.2f}%"], "Status": [result]}
                st.table(data)

    # Section 3: Log Section
    elif page == "About":
        st.header("Log Section")
        st.write("System logs will be displayed here.")

if __name__ == "__main__":
    main()