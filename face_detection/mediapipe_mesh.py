import cv2
import time

import mediapipe as mp

"""
This module provides functions to perform face mesh detection with Mediapipe.

Author: Code mostly used from https://www.computervision.zone/courses/advance-computer-vision-with-python/
Date: May 28, 2023
"""


class FaceMeshDetector:
    def __init__(
        self,
        staticMode=False,
        maxFaces=2,
        redefineLms=False,
        minDetectionCon=0.5,
        minTrackCon=0.5,
    ):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.redefineLms = redefineLms
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        # Face mesh model
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(
            color=(255, 255, 255), thickness=2, circle_radius=1
        )
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            self.staticMode,
            self.maxFaces,
            self.redefineLms,
            self.minDetectionCon,
            self.minTrackCon,
        )

    def findFaceMesh(self, img, draw=True):
        # RGB conversion
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detection
        self.results = self.faceMesh.process(self.imgRGB)

        faces = []
        if self.results.multi_face_landmarks:
            # Draw a face mesh for each detected face
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img,
                        faceLms,
                        self.mpFaceMesh.FACEMESH_CONTOURS,
                        self.drawSpec,
                        self.drawSpec,
                    )
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # Show id numbers for the landmarks
                    # cv2.putText(
                    #    img,
                    #    str(id),
                    #    (x, y),
                    #    cv2.FONT_HERSHEY_SIMPLEX,
                    #    2,
                    #    (255, 255, 255),
                    #    1,
                    # )
                    # Append x and y coordinates for each landmark
                    face.append([x, y])
                faces.append(face)
        return img, faces


def main():
    # Specify your path to your video file here
    video_path = (
        "C:\\Users\\AndreiBorg\\stroke-extra\\Facialispares_Baseline_Viktor.MP4"
    )

    # Use video_path or 0 for webcam
    cap = cv2.VideoCapture(video_path)

    pTime = 0
    detector = FaceMeshDetector()

    # Loop through each video frame
    while True:
        success, img = cap.read()

        # Stop the program if video ends or a frame cannot be read
        if not success:
            break

        img, faces = detector.findFaceMesh(img)

        # Fps counter
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(
            img,
            f"MediaPipe",
            (680, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            4,
            (0, 90, 255),
            8,
        )

        ## Create named window and set window size
        # cv2.namedWindow("face_cam", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("face_cam", 1400, 800)
        #
        # cv2.imshow("face_cam", img)

        # Exit if the 'q' key is pressed, use waitKey(1) for fastest fps
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quitting video...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
