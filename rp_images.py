import cv2
import os
import mediapipe as mp
import time
import numpy as np
import math
import matplotlib.pyplot as plt
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

    def findFaceMesh(self, img, euc_dist_left, euc_dist_right, draw=True):
        # RGB conversion
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detection
        self.results = self.faceMesh.process(self.imgRGB)

        faces = []
        if self.results.multi_face_landmarks:
            # Draw a face mesh for each detected face
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    face = []
                    custom_points = [93, 323, 57, 291]
                    for p in custom_points:
                        ih, iw, ic = img.shape
                        x, y = int(faceLms.landmark[p].x * iw), int(
                            faceLms.landmark[p].y * ih
                        )
                        if p == 93:
                            left_ear = [x, y]
                        if p == 323:
                            right_ear = [x, y]
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
                            # print("EUC distance: ", euc_dist)
                            euc_dist_left.append(math.dist([x, y], left_ear))
                            # print ("List:", euc_dists)
                        if p == 291:
                            euc_dist_right.append(math.dist([x, y], right_ear))
                            # print ("List:", euc_dist_right)
                faces.append(face)
        return img, faces, euc_dist_left, euc_dist_right


def main():
    # Specify your path to your video file here + other ariables for quick editing
    video_path = "D:\\Kandidatarbete\\extra2\\s10.mov"
    save_name = "D:\\Kandidatarbete\\denna2\\weak19"
    # max_frame_length = 86

    # Use video_path or 0 for webcam
    cap = cv2.VideoCapture(video_path)

    pTime = 0
    detector = FaceMeshDetector()

    euc_distance_left = []
    euc_distance_right = []

    # Loop through each video frame
    # for i in range(max_frame_length):
    while True:
        success, img = cap.read()

        # Stop the program if video ends or a frame cannot be read
        if not success:
            break

        img, faces, euc_l, euc_r = detector.findFaceMesh(
            img, euc_distance_left, euc_distance_right
        )

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
    euc_l = np.array(euc_l)
    euc_r = np.array(euc_r)

    # Recurrence plot transformation
    # Convert the time series data to recurrence plots with the pyts library
    rp_l = RecurrencePlot(threshold="point", percentage=20).fit_transform(
        euc_l.reshape(1, -1)
    )[0]
    rp_r = RecurrencePlot(threshold="point", percentage=20).fit_transform(
        euc_r.reshape(1, -1)
    )[0]

    # Concatenate the two recurrence plots horizontally
    concatenated_rp = np.concatenate((rp_l, rp_r), axis=1)

    # Create a figure with three subplots
    # Create a figure and plot the concatenated recurrence plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")  # Turn off axis
    ax.imshow(concatenated_rp, cmap="gray")
    # ax.set_title('Concatenated Recurrence Plot')
    ax.invert_yaxis()  # Flip the y-axis

    # Adjust the layout and save the figure with a specific size
    fig.set_size_inches(4, 4)
    plt.savefig(save_name, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
