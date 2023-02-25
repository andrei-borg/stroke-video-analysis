import cv2
import torch
from retinaface import RetinaFace

# Load the RetinaFace model (didn't work)
#detector = RetinaFace()

# Specify the input video file path
video_path = 'C:\\Users\\AndreiBorg\\Dolleyes4K.mp4'

# Create a VideoCapture object to read frames from the video
cap = cv2.VideoCapture(video_path)

# Loop over the frames in the video
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    # Stop if end of the video
    if not ret:
        break
    
    # Perform face detection using the RetinaFace model
    faces = RetinaFace.detect_faces(frame)
    
    # Draw bounding boxes on the faces in the frame
    for face in faces:
        x1, y1, x2, y2, score = face['box']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
