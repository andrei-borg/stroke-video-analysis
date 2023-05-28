import cv2
import math

import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

from pyts.image import RecurrencePlot

"""
This file creates recurrence plots for a given image directory.

Author: Andrei Borg
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
            color=(255, 255, 255), thickness=1, circle_radius=1
        )
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            self.staticMode,
            self.maxFaces,
            self.redefineLms,
            self.minDetectionCon,
            self.minTrackCon,
        )

    def findFaceMesh(
        self,
        img,
        euc_dist_left_mouth,
        euc_dist_right_mouth,
        euc_dist_left_eye,
        euc_dist_right_eye,
        draw=True,
    ):
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
                    # Point indexes for left and right corner of the mouth, ears, and eye lids (top and bottom)
                    custom_points = [93, 323, 57, 291, 159, 23, 385, 253]
                    for p in custom_points:
                        ih, iw, ic = img.shape
                        x, y = int(faceLms.landmark[p].x * iw), int(
                            faceLms.landmark[p].y * ih
                        )
                        if p == 93:
                            left_ear = [x, y]
                        if p == 323:
                            right_ear = [x, y]
                        if p == 159:
                            left_upper_eye = [x, y]
                        if p == 385:
                            right_upper_eye = [x, y]

                        # Show id numbers for the landmarks
                        cv2.putText(
                            img,
                            str(p),
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.15,
                            # Color of the landmarks
                            (255, 255, 255),
                            1,
                        )
                        # Append x and y coordinates for each landmark
                        face.append([x, y])
                        # Calculate the eucledian distance between the landmarks
                        if p == 57:
                            euc_dist_left_mouth.append(math.dist([x, y], left_ear))
                        if p == 291:
                            euc_dist_right_mouth.append(math.dist([x, y], right_ear))
                        if p == 23:
                            euc_dist_left_eye.append(math.dist([x, y], left_upper_eye))
                        if p == 253:
                            euc_dist_right_eye.append(
                                math.dist([x, y], right_upper_eye)
                            )
        return (
            img,
            faces,
            euc_dist_left_mouth,
            euc_dist_right_mouth,
            euc_dist_left_eye,
            euc_dist_right_eye,
        )


def main():
    ## ----- Mouth ----- ##

    # Specify your path to your video file here
    video_path = "D:\\Kandidatarbete\\Dataset Face\\stroke\\Mouth\\Andrei_wL1.mp4"
    save_name = "D:\\Kandidatarbete\\new\\mouth.png"

    # Use video_path or 0 for webcam
    cap = cv2.VideoCapture(video_path)

    detector = FaceMeshDetector()

    euc_distance_left_mouth = []
    euc_distance_right_moth = []
    euc_distance_left_eye = []
    euc_distance_right_eye = []

    # Loop through each video frame
    while True:
        success, img = cap.read()

        # Stop the program if video ends or a frame cannot be read
        if not success:
            break

        img, faces, euc_lm, euc_rm, euc_le, euc_re = detector.findFaceMesh(
            img,
            euc_distance_left_mouth,
            euc_distance_right_moth,
            euc_distance_left_eye,
            euc_distance_right_eye,
        )

        cv2.imshow("face_cam", img)

        # Exit if the "q" key is pressed, use waitKey(1) for fastest fps
        if cv2.waitKey(2) & 0xFF == ord("q"):
            print("Quitting video...")
            break

    cap.release()
    cv2.destroyAllWindows()

    ## Recurrence plot ##
    euc_lm = np.array(euc_lm)
    euc_rm = np.array(euc_rm)

    # Recurrence plot transformation
    # Convert the time series data to recurrence plots with the pyts library
    rp_lm = RecurrencePlot(threshold="point", percentage=20).fit_transform(
        euc_lm.reshape(1, -1)
    )[0]
    rp_rm = RecurrencePlot(threshold="point", percentage=20).fit_transform(
        euc_rm.reshape(1, -1)
    )[0]

    # Concatenate the two recurrence plots horizontally
    concatenated_rp = np.concatenate((rp_lm, rp_rm), axis=1)

    # Create a figure with three subplots
    # NOTE: This is for demonstration purposes only, go to the next "NOTE" section for the actual plotting

    # fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

    ## Plot the first recurrence plot in the first subplot
    # axs[0].imshow(rp_lm, cmap="gray")
    # axs[0].set_title("Left")
    # axs[0].axis("off")
    # axs[0].invert_yaxis()

    ## Plot the second recurrence plot in the second subplot
    # axs[1].imshow(rp_rm, cmap="gray")
    # axs[1].set_title("Right")
    # axs[1].axis("off")
    # axs[1].invert_yaxis()

    ## Plot the concatenated recurrence plot in the third subplot
    # axs[2].imshow(concatenated_rp, cmap="gray")
    # axs[2].set_title("Kombinerade Recurrence Plots")
    # axs[2].axis("off")
    # axs[2].invert_yaxis()

    ## Adjust the layout and save the figure with a specific size
    # fig.set_size_inches(10, 5)
    # plt.savefig("recurrence_plots.png", bbox_inches="tight", pad_inches=0, dpi=300)
    # plt.show()

    # Create a figure and plot the concatenated recurrence plot
    # NOTE: This is used as images data for the neural networks
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")  # Turn off axis
    ax.imshow(concatenated_rp, cmap="gray")
    # ax.set_title('Concatenated Recurrence Plot')
    ax.invert_yaxis()  # Flip the y-axis

    # Adjust the layout and save the figure with a specific size
    fig.set_size_inches(1, 1)
    plt.savefig(save_name, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()

    ## ----- Eyes ----- ##

    # Specify your path to your video file here + other ariables for quick editing
    video_path2 = "D:\\Kandidatarbete\\Dataset Face\\stroke\\Eyes\\Facialispares 3 - Marcus - 1.mp4"
    save_name2 = "D:\\Kandidatarbete\\new\\eye.png"

    # Use video_path or 0 for webcam
    cap = cv2.VideoCapture(video_path2)

    detector = FaceMeshDetector()

    euc_distance_left_mouth = []
    euc_distance_right_moth = []
    euc_distance_left_eye = []
    euc_distance_right_eye = []

    # Loop through each video frame
    while True:
        success, img = cap.read()

        # Stop the program if video ends or a frame cannot be read
        if not success:
            break

        img, faces, euc_lm, euc_rm, euc_le, euc_re = detector.findFaceMesh(
            img,
            euc_distance_left_mouth,
            euc_distance_right_moth,
            euc_distance_left_eye,
            euc_distance_right_eye,
        )

        cv2.imshow("face_cam", img)

        # Exit if the "q" key is pressed, use waitKey(1) for fastest fps
        if cv2.waitKey(2) & 0xFF == ord("q"):
            print("Quitting video...")
            break

    ## Recurrence plot ##
    euc_le = np.array(euc_le)
    euc_re = np.array(euc_re)

    # Recurrence plot transformation
    # Convert the time series data to recurrence plots with the pyts library
    rp_le = RecurrencePlot(threshold="point", percentage=20).fit_transform(
        euc_le.reshape(1, -1)
    )[0]
    rp_re = RecurrencePlot(threshold="point", percentage=20).fit_transform(
        euc_re.reshape(1, -1)
    )[0]

    # Concatenate the two recurrence plots horizontally
    concatenated_rp = np.concatenate((rp_le, rp_re), axis=1)

    # Create a figure and plot the concatenated recurrence plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")  # Turn off axis
    ax.imshow(concatenated_rp, cmap="gray")
    # ax.set_title('Concatenated Recurrence Plot')
    ax.invert_yaxis()  # Flip the y-axis

    # Adjust the layout and save the figure with a specific size
    fig.set_size_inches(2, 2)
    plt.savefig(save_name2, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
