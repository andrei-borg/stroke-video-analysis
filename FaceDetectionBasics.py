import cv2
import time
from retinaface import RetinaFace
from mtcnn import MTCNN

#video_path = '/Users/andreiborg/Dolleyes4K.mp4'

cap = cv2.VideoCapture(0)
pTime = 0

detector = MTCNN()

while True:
    success, img = cap.read()

    faces = detector.detect_faces(img)

    for face in faces:
        bbox = faces[0]['box']
        confidence = faces[0]['confidence']
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 255, 255), 2)
    
        # Add face label
        label = f"Confidence {confidence:.2f}"
        cv2.putText(img, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
    cv2.imshow('face_cam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()