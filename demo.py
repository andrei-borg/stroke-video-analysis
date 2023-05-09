import cv2
import os
import mediapipe as mp
import time
import numpy as np
import math
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPool2D,
    Flatten,
    Dense,
)
from PIL import Image
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
                            (255, 255, 255),
                            1,
                        )
                        # Append x and y coordinates for each landmark
                        face.append([x, y])
                        # Calculate the eucledian distance between the landmarks
                        if p == 57:
                            # print("EUC distance: ", euc_dist)
                            euc_dist_left_mouth.append(math.dist([x, y], left_ear))
                            # print ("List:", euc_dists)
                        if p == 291:
                            euc_dist_right_mouth.append(math.dist([x, y], right_ear))
                            # print ("List:", euc_dist_right_mouth)
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
    # Mouth
    img_height_mouth = 465
    img_width_mouth = 930

    # Specify your path to your video file here + other ariables for quick editing
    #video_path = "D:\\Kandidatarbete\\Dataset Face\\stroke\\Mouth\\Andrei_wL1.mp4"
    #save_name = "D:\\Kandidatarbete\\output\\mouth.png"
    save_name = "/Volumes/ANDREI 1 TB/Kandidatarbete/output/mouth.png"
    # max_frame_length = 86

    # Use video_path or 0 for webcam
    cap = cv2.VideoCapture(0)

    detector = FaceMeshDetector()

    euc_distance_left_mouth = []
    euc_distance_right_moth = []
    euc_distance_left_eye = []
    euc_distance_right_eye = []

    # Loop through each video frame
    # for i in range(max_frame_length):
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

        # Exit if the 'q' key is pressed, use waitKey(1) for fastest fps
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

    ## Create a figure and plot the concatenated recurrence plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")  # Turn off axis
    ax.imshow(concatenated_rp, cmap="gray")

    # Adjust the layout and save the figure with a specific size
    fig.set_size_inches(4, 4)
    plt.savefig(save_name, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()

    model_test = Sequential()
    model_test.add(
        Conv2D(
            filters=256,
            kernel_size=(6, 6),
            activation="relu",
            padding="same",
            input_shape=(img_height_mouth, img_width_mouth, 3),
        )
    )
    model_test.add(MaxPool2D(pool_size=(4, 4), strides=4))

    model_test.add(
        Conv2D(filters=512, kernel_size=(9, 9), activation="relu", padding="same")
    )
    model_test.add(MaxPool2D(pool_size=(4, 4), strides=4))
    model_test.add(
        Conv2D(filters=512, kernel_size=(9, 9), activation="relu", padding="same")
    )
    model_test.add(MaxPool2D(pool_size=(4, 4), strides=4))

    model_test.add(Flatten())
    model_test.add(Dense(512, activation="relu"))
    model_test.add(Dense(units=2, activation="softmax"))

    # Load the weights from the saved file
    model_test.load_weights("mouth_weights_10ep.h5")

    # Load the image and preprocess it
    img = Image.open(save_name)
    # img = Image.open("D:\\Kandidatarbete\\rp_eyes\\test\\normal\\normal28.png")

    img = img.resize((930, 465))  # Mouth
    # img = img.resize((697, 348)) # Eyes

    # Remove the alpha channel if present
    if img.mode == "RGBA":
        img = img.convert("RGB")

    # Convert the image to a numpy array
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Use the model to predict on a new image
    predictions = model_test.predict(img_array)

    # Get the confidence score for the positive class
    confidence_score_normal = predictions[0][0]
    confidence_score_stroke = predictions[0][1]

    print("Confidence score for normal:", confidence_score_normal)
    print("Confidence score for stroke:", confidence_score_stroke)


if __name__ == "__main__":
    main()
