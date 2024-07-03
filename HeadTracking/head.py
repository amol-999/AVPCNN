import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Initialize MediaPipe Drawing utility
mp_drawing = mp.solutions.drawing_utils

# Directory containing frames
input_dir = 'DataCollection/extracted_frames'

# Initialize frame number
frame_number = 0

for frame_filename in sorted(os.listdir(input_dir)):
    frame = cv2.imread(os.path.join(input_dir, frame_filename))

    # Convert the BGR image to RGB before processing
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Draw face detections of each face
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)
            print(f"Face detected on {frame_filename}!")

    # Process the RGB image for face landmarks
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Draw face landmarks of each face
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACE_CONNECTIONS)
            print(f"Face landmarks detected on frame number {frame_filename}!")

    # Show the image
    cv2.imshow('MediaPipe Face Detection', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    # Increment frame number
    frame_number += 1

cv2.destroyAllWindows()
