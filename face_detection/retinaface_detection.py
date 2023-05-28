import cv2
import time
import torch

import face_detection as fd

"""
This module provides functions to perform face detection with Retinaface.

Author: Andrei Borg
Date: May 28, 2023
"""


class FaceDetector(object):
    def __init__(
        self,
        name="RetinaNetResNet50",
        confidence_threshold=0.5,
        nms_iou_threshold=0.3,
        device=0,
    ):
        self.name = name
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = device

        # Face detection model
        self.faceDetection = fd.build_detector(
            self.name, self.confidence_threshold, self.nms_iou_threshold, self.device
        )

    def findFaces(self, img, draw=True):
        # RGB conversion
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detection
        self.results = self.faceDetection.detect(self.imgRGB)

        bboxs = []
        if len(self.results) > 0:
            # Draw a bounding box for each detected face
            for id, detection in enumerate(self.results):
                bbox = detection
                score = detection[4]

                bbox[2] = bbox[2] - bbox[0]
                bbox[3] = bbox[3] - bbox[1]

                bbox = [int(i) for i in bbox[:4]]

                # Append bbox coordinates & score to bboxs
                bboxs.append([id, bbox, score])

                if draw:
                    img = self.fancyDraw(img, bbox)
                    # Detection score
                    cv2.putText(
                        img,
                        f"{int(score*100)}%",
                        (bbox[0], bbox[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (255, 0, 255),
                        3,
                    )
        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=12, rt=2):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        white = (255, 255, 255)

        # Top left corner
        cv2.rectangle(img, bbox, white, rt)
        cv2.line((img), (x, y), (x + l, y), white, t)
        cv2.line((img), (x, y), (x, y + l), white, t)

        # Top right corner
        cv2.rectangle(img, bbox, white, rt)
        cv2.line((img), (x1, y), (x1 - l, y), white, t)
        cv2.line((img), (x1, y), (x1, y + l), white, t)

        # Bottom left corner
        cv2.rectangle(img, bbox, white, rt)
        cv2.line((img), (x, y1), (x + l, y1), white, t)
        cv2.line((img), (x, y1), (x, y1 - l), white, t)

        # Bottom right corner
        cv2.rectangle(img, bbox, white, rt)
        cv2.line((img), (x1, y1), (x1 - l, y1), white, t)
        cv2.line((img), (x1, y1), (x1, y1 - l), white, t)

        return img


def main():
    # Specify your path to your video file here
    video_path = (
        "C:\\Users\\AndreiBorg\\stroke-extra\\Facialispares_Baseline_Viktor.MP4"
    )

    # Use video_path or 0 for webcam
    cap = cv2.VideoCapture(video_path)

    pTime = 0
    detector = FaceDetector()

    if torch.cuda.is_available():
        print("CUDA is available!")
    else:
        print("CUDA is not available!")

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
            f"RetinaFace",
            (640, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            4,
            (210, 255, 0),
            8,
        )
        cv2.imshow("face_cam", img)

        # Exit if the "q" key is pressed, use waitKey(1) for fastest fps
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quitting video...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
