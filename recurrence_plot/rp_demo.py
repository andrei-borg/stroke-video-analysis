import cv2

import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pyts.image import RecurrencePlot

from rp_generation import FaceMeshDetector

from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPool2D,
    Flatten,
    Dense,
)

"""
This file demonstrates how a live webcam recording can be used
to classify a face expression as "non-stroke" or "stroke". 
The recurrence plot method is used, but Eigenface or HOG would also 
work and most likely give better predictions.
This code analyzes the mouth only, the eyes aren't considered.

Instructions:   

- Run the code and wait for the webcam to appear.
- Try not to move the head while doing a stroke or non-stroke face, the algorithm is sensitive to fast head movements.
- Sit straight in front of the camera and do the expression for 2-3 seconds. Then hit the "Q" button on the keyboard to end the recording.
- NOTE: When doing the expression -> start from a relaxed position, quickly do the desired expression, return to the relaxed position, press "Q".
- Confidence scores should appear in the terminal, displaying the corresponding scores.

Author: Andrei Borg
Date: May 28, 2023
"""


def main():
    # Recurrance plot image dimensions
    img_height_mouth = 465
    img_width_mouth = 930

    # Specify your path where the generated recurrence plot will be placed
    save_name = "D:\Kandidatarbete\output\mouth_plot.png"

    # Use video_path or 0 for webcam
    cap = cv2.VideoCapture(0)

    detector = FaceMeshDetector()

    # In this version only the mouth coordinates will be used
    left_mouth = []
    right_moth = []
    left_eye = []
    right_eye = []

    # Loop through each video frame
    while True:
        success, img = cap.read()

        # Stop the program if a frame can't be read
        if not success:
            break

        img, faces, euc_lm, euc_rm, euc_le, euc_re = detector.findFaceMesh(
            img,
            left_mouth,
            right_moth,
            left_eye,
            right_eye,
        )

        cv2.imshow("face_cam", img)

        # Exit if the "q" key is pressed, use waitKey(1) for fastest fps
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quitting video...")
            break

    cap.release()
    cv2.destroyAllWindows()

    ## Recurrence plot ##

    print("Analysing the video...")
    euc_lm = np.array(euc_lm)
    euc_rm = np.array(euc_rm)

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

    # CNN model
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

    # Load weights from a trained CNN model with the same structure as above
    model_test.load_weights("mouth_weights_10ep.h5")

    # Load the image and preprocess it
    img = Image.open(save_name)

    img = img.resize((img_width_mouth, img_height_mouth))

    # Remove the alpha channel if present
    if img.mode == "RGBA":
        img = img.convert("RGB")

    # Convert the image to a numpy array
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Use the model to make a prediction on the generated recurrence plot
    predictions = model_test.predict(img_array)

    # Get the confidence score for the two classes
    confidence_score_normal = predictions[0][0]
    confidence_score_stroke = predictions[0][1]

    # Percentage conversion
    percentage_normal = confidence_score_normal * 100
    percentage_stroke = confidence_score_stroke * 100

    print(f"Confidence score for normal: {percentage_normal:.2f}%")
    print(f"Confidence score for stroke: {percentage_stroke:.2f}%")


if __name__ == "__main__":
    main()
