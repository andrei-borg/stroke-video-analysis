from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

# Read the image
img = imread("C:\\Users\\AndreiBorg\\stroke-extra\\AndreiHOG.jpg")

# Create hog features
fd, hog_img = hog(
    img,
    orientations=8,
    pixels_per_cell=(16, 16),
    cells_per_block=(1, 1),
    visualize=True,
    channel_axis=-1,
)

# Figures
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

# Input image
ax1.axis("off")
ax1.imshow(img, cmap="gray")
ax1.set_title("Input image")

# Rescale the histogram for better display
hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))

# Output HoG image
ax2.axis("off")
ax2.imshow(hog_img_rescaled, cmap="gray")
ax2.set_title("Histogram of Oriented Gradients")
plt.show()
