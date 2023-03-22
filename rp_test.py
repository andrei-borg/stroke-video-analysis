import cv2
import mediapipe as mp
import time
from math import dist
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import RecurrencePlot


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
            color=(255, 255, 0), thickness=1, circle_radius=1
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
        euc_distance_left = []
        if self.results.multi_face_landmarks:
            # Draw a face mesh for each detected face
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    face = []
                    custom_points = [1, 57, 291]
                    for p in custom_points:
                        ih, iw, ic = img.shape
                        x, y = int(faceLms.landmark[p].x * iw), int(
                            faceLms.landmark[p].y * ih
                        )
                        if p == 1:
                            nose_coords = [x, y]
                        # Show id numbers for the landmarks
                        cv2.putText(
                            img,
                            str(p),
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.25,
                            (255, 255, 0),
                            1,
                        )
                        # Append x and y coordinates for each landmark
                        face.append([x, y])
                        # Calculate the eucledian distance between the landmarks
                        if p == 57:
                            euc_dist = dist([x, y], nose_coords)
                            # print("EUC distance: ", euc_dist)
                            euc_distance_left.append(euc_dist)
                faces.append(face)
        return img, faces, euc_distance_left


def main():
    # Specify your path to your video file here
    video_path = "C:\\Users\\AndreiBorg\\stroke-extra\\C0016.MP4"

    # Use video_path or 0 for webcam
    cap = cv2.VideoCapture(video_path)

    pTime = 0
    detector = FaceMeshDetector()

    # Loop through each video frame
    while True:
        success, img = cap.read()
        img, faces, euc_dist = detector.findFaceMesh(img)

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
        if cv2.waitKey(2) & 0xFF == ord("q"):
            print("Quitting video...")
            break

    cap.release()
    cv2.destroyAllWindows()

    ## Recurrence plot ##

    # Get the recurrence plots for all the time series
    rp = RecurrencePlot(threshold="point", percentage=20)
    X_rp = rp.fit_transform(euc_dist)

    # Plot the recurrence plot
    fig = plt.figure()
    plt.imshow(X_rp, cmap="binary")

    plt.show()


if __name__ == "__main__":
    main()
