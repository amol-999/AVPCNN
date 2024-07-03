import cv2
import dlib
import numpy as np
import os

# Initialize dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()

# Get the directory of the current script
script_dir = 'DataCollection/extracted_frames'
predictor_path = os.path.join(script_dir, 'shape_predictor_68_face_landmarks.dat')

predictor = dlib.shape_predictor(predictor_path)

def get_head_pose(shape):
    # 2D image points from 68 facial landmarks
    image_points = np.array([
        (shape[33, :]),     # Nose tip
        (shape[8, :]),      # Chin
        (shape[36, :]),     # Left eye left corner
        (shape[45, :]),     # Right eye right corner
        (shape[48, :]),     # Left mouth corner
        (shape[54, :])      # Right mouth corner
    ], dtype="double")

    # 3D model points
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # Camera internals
    focal_length = shape[0, 1]
    center = (shape[0, 0]/2, shape[0, 1]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    return (rotation_vector, translation_vector)

cap = cv2.VideoCapture('./DataCollection/test_vid.MP4')


frame_count = 0
frames_to_read = True
while frames_to_read:
    ret, frame = cap.read()
    if ret:
        print(frame.dtype)
        print(frame)
    #cv2.imshow('Frame', frame)
    if frame is None:
        print("No frame available. Check the video file.")
        continue
    if not ret:
        break
    if not ret or frame is None:
        print("No frame available. Check the video file.")
        continue
    cv2.imshow('Frame', frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(rgb)
    rgb = cv2.convertScaleAbs(rgb)
    try:
        rects = detector(rgb, 0)
        for rect in rects:
            shape = predictor(rgb, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            # Draw facial landmarks
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            (rotation_vector, translation_vector) = get_head_pose(shape)

            # Project a 3D point (0, 0, 1000.0) onto the image plane to draw a line from the nose towards the direction the person is looking
            (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            # Draw the line
            p1 = (int(shape[33, 0]), int(shape[33, 1]))
            p2 = (int(nose_end_point2D[0, 0, 0]), int(nose_end_point2D[0, 0, 1]))

            cv2.line(frame, p1, p2, (255, 0, 0), 2)

            print(f"Face detected in frame {frame_count}")
        cv2.imshow("Output", frame)
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
        frame_count += 1
    except RuntimeError as e:
        print(f"An error occurred while processing the frame: {e}")    
    finally:
        frames_to_read = False    
        cap.release()
        cv2.destroyAllWindows()    

#convert to for loop iterating over array of file locations.
#create an empty array
#populate the array with extracted_frames
#traverse the end of the array     