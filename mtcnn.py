import cv2
from mtcnn import MTCNN

# Load the MTCNN model
detector = MTCNN()

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
    
    # Perform face detection using the MTCNN model
    results = detector.detect_faces(frame)
    
    # Draw bounding boxes on the faces in the frame
    for result in results:
        x, y, width, height = result['box']
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
