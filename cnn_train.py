import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# Define constants
HEIGHT = 465
WIDTH = 930
#IMG_SIZE = 128
NUM_CHANNELS = 1  # grayscale
DATA_DIR = 'data'

# Load and preprocess images
X = []
y = []

for label, folder_name in enumerate(['negative', 'positive']):
    folder_path = os.path.join(DATA_DIR, folder_name)
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        #img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.reshape((HEIGHT, WIDTH, NUM_CHANNELS))
        img = img.astype('float32') / 255.0
        X.append(img)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the shape of each input plot
plot_shape = (HEIGHT, WIDTH, NUM_CHANNELS)

# Define the CNN architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=plot_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])