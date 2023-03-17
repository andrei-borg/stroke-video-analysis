import cv2
import dlib
import time


class FaceDetector(object):
    def __init__(self):
        # Face detection model
        self.faceDetection = dlib.get_frontal_face_detector()

    def findFaces(self, img, draw=True):
        # RGB conversion
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detection
        self.results = self.faceDetection(self.imgRGB)

        bboxs = []
        if len(self.results) > 0:
            # Draw a bounding box for each detected face
            for id, det in enumerate(self.results):
                bbox = [
                    det.left(),
                    det.top(),
                    det.right() - det.left(),
                    det.bottom() - det.top(),
                ]

                # Append bbox coordinates to bboxs
                bboxs.append([id, bbox])

                if draw:
                    img = self.fancyDraw(img, bbox)
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
    cap = cv2.VideoCapture(video_path)

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
            f"Dlib HOG - FPS: {int(fps)}",
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
