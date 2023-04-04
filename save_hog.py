
# A script that create HOG-images based on all images in a map

import os
import cv2
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

# Folder with images from the video
folder1 = "/Users/elsathorestrom/Documents/Chalmers/Kanditatarbete/Egen_programmering/frames_facialispares_3/"

# A function that iterates over all images in a folder and saves them as HOG-images
def save_hog(folder):
    # Iterate over all images stored from the video
    for filename in os.listdir(folder):
        print(filename)
        img = imread(f"/Users/elsathorestrom/Documents/Chalmers/Kanditatarbete/Egen_programmering/frames_facialispares_3/{filename}")
        # Create HOG-features
        fd, hog_img = hog(
        img,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=True,
        channel_axis=-1,
        )

        hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))
        # plt.imshow(hog_img_rescaled, cmap = "gray")
        # Save HOG-image in a new folder
        plt.imsave(f'/Users/elsathorestrom/Documents/Chalmers/Kanditatarbete/Egen_programmering/hog_facialispares/hog_{filename}.jpg', hog_img_rescaled, cmap = "gray")

save_hog(folder1)