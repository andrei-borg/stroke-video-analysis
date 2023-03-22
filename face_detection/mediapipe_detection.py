import cv2
import time
import mediapipe as mp


class FaceDetector(object):
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon

        # Face detection model
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        # RGB conversion
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detection
        self.results = self.faceDetection.process(self.imgRGB)

        bboxs = []
        if self.results.detections:
            # Draw a bounding box for each detected face
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )
                # Append bbox coordinates & score to bboxs
                bboxs.append([id, bbox, detection.score])

                if draw:
                    img = self.fancyDraw(img, bbox)
                    # Detection score
                    cv2.putText(
                        img,
                        f"{int(detection.score[0]*100)}%",
                        (bbox[0], bbox[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (255, 0, 255),
                        3,
                    )
        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        # Top left corner
        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        cv2.line((img), (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line((img), (x, y), (x, y + l), (255, 0, 255), t)

        # Top right corner
        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        cv2.line((img), (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line((img), (x1, y), (x1, y + l), (255, 0, 255), t)

        # Bottom left corner
        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        cv2.line((img), (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line((img), (x, y1), (x, y1 - l), (255, 0, 255), t)

        # Bottom right corner
        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        cv2.line((img), (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line((img), (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        return img


def main():
    # Specify your path to your video file here
    video_path = "C:\\Users\\AndreiBorg\\stroke-extra\\C0016.MP4"

    # Use video_path or 0 for webcam
    cap = cv2.VideoCapture(0)

    pTime = 0
    detector = FaceDetector()

    # Loop through each video frame
    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)

        # Stop the program if video ends or a frame cannot be read
        if not success:
            break

        # Fps counter
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(
            img,
            f"MediaPipe - FPS: {int(fps)}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 255, 0),
            3,
        )
        cv2.imshow("face_cam", img)

        # Exit if the 'q' key is pressed, use waitKey(1) for fastest fps
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quitting video...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
