import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

gen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    zoom_range=0.1,
    horizontal_flip=True,
)

image_path = (
    "rp_data/test/stroke/Viktor_weakR4.png"  # Change this target image to augment
)
image = np.expand_dims(plt.imread(image_path), 0)

aug_iter = gen.flow(
    image,
    save_to_dir="rp_data/test/stroke",
    save_prefix="aug",
    save_format="png",
)

aug_images = [
    next(aug_iter)[0].astype(np.uint8)
    for i in range(7)  # Change range(x) for generating x augmentations
]

print("Created augmented images for:", image_path)
