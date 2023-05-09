import cv2
import os
import mediapipe as mp
import time
import numpy as np
import math
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
from rp_images import FaceMeshDetector
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPool2D,
    Flatten,
    Dense,
)
from PIL import Image


def main():
    # Mouth
    img_height_mouth = 465
    img_width_mouth = 930

    # Specify your path to your video file here + other ariables for quick editing
    video_path = "D:\\Kandidatarbete\\Dataset Face\\stroke\\Mouth\\Andrei_wL1.mp4"
    save_name = "D:\\Kandidatarbete\\output\\mouth.png"
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
