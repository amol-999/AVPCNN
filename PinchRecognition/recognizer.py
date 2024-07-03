# Media pipe based gesture recognition script 
# A Michael Lance & Amol Gupta experience
# 6/27/2024
# ------------------------------------------------------------------------------------------------------------------------#

import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
import cv2

# path to the pretrained model itself, courtesy of Google
model_path = "./PinchRecognition/gesture_recognizer.task"

DESIRED_WIDTH = 480
DESIRED_HEIGHT = 480

# instance everything from media pipe that we will need to interact with the model
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

image_file_names = []
images = []
results = []

for filename in os.listdir("DataCollection/extracted_frames"):
    image_file_names.append("DataCollection/extracted_frames/" + filename)

for image_file_name in image_file_names:
    try:
        image = mp.Image.create_from_file(image_file_name)
        recognition_result = recognizer.recognize(image)

        # Check if gestures list is not empty
        if recognition_result.gestures:
            top_gesture = recognition_result.gestures[0][0]
            hand_landmarks = recognition_result.hand_landmarks
            results.append((top_gesture, hand_landmarks))
            print(f"Recognized gesture {top_gesture} in image {image_file_name}")
        else:
            print(f"No gestures recognized in image {image_file_name}")
    except Exception as e:
        print(f"An error occurred while processing image {image_file_name}: {str(e)}")

def display_batch_of_images_with_gestures_and_hand_landmarks(images, results):
    for i, (image, result) in enumerate(zip(images, results)):
        top_gesture, hand_landmarks = result

        # Convert the image from MediaPipe Image format to a format that can be used with matplotlib
        image_rgb = cv2.cvtColor(image.get_image_data_numpy(), cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        plt.title(f"Image {i}: Top gesture = {top_gesture}")

        # Assuming hand_landmarks is a list of (x, y) tuples
        for landmark in hand_landmarks:
            plt.scatter(*landmark, c='r')
         # Display the top recognized gesture
        print(f"Top recognized gesture in image {i}: {top_gesture}")    

        plt.show()

display_batch_of_images_with_gestures_and_hand_landmarks(images, results)
