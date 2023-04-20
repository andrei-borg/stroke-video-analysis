import numpy as np
# import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

gen = ImageDataGenerator(
    rotation_range=14,
    width_shift_range=0.08,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.07,
    horizontal_flip=True,
)

for filename in os.listdir("rp_data/valid/normal"):
    if 'normal' in filename:

        image_path = "rp_data/valid/normal/" + filename
        
        image = np.expand_dims(plt.imread(image_path), 0)

        aug_iter = gen.flow(
            image,
            save_to_dir="rp_data/valid/normal",
            save_prefix="new",
            save_format="png",
        )

        aug_images = [
            next(aug_iter)[0].astype(np.uint8)
            for i in range(50)  # Change range(x) for generating x augmentations
        ]

        print("Created augmented images for:", filename)
    