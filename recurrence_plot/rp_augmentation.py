import os

import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

"""
This file creates augmented images for a given directory.

Author: Andrei Borg
Date: May 28, 2023
"""


def main():
    gen = ImageDataGenerator(
        rotation_range=14,
        width_shift_range=0.08,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.07,
        horizontal_flip=True,
    )

    path = "D:\\Kandidatarbete\\rp_eyes\\train\\stroke"
    for filename in os.listdir(path):
        if "stroken" in filename:
            image_path = path + "\\" + filename

            image = np.expand_dims(plt.imread(image_path), 0)

            aug_iter = gen.flow(
                image,
                save_to_dir=path,
                save_prefix="aug",
                save_format="png",
            )

            aug_images = [
                next(aug_iter)[0].astype(np.uint8)
                for i in range(20)  # Change range(x) for generating x augmentations
            ]

            print("Created augmented images for:", filename)


if __name__ == "__main__":
    main()
