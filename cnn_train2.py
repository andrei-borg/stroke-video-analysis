import os
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# Define image size and number of channels
IMG_HEIGHT = 465
IMG_WIDTH = 930
NUM_CHANNELS = 1

# Define the directories where the positive and negative images are stored
positive_dir = "data/positive"
negative_dir = "data/negative"

# Create empty lists to store the image data and labels
data = []
labels = []

# Load the positive images and append to the data list with label 1
for filename in os.listdir(positive_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(positive_dir, filename), cv2.IMREAD_GRAYSCALE)
        img = img.reshape((IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
        data.append(img)
        labels.append(1)

# Load the negative images and append to the data list with label 0
for filename in os.listdir(negative_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(negative_dir, filename), cv2.IMREAD_GRAYSCALE)
        img = img.reshape((IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
        data.append(img)
        labels.append(0)

# Convert the data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Split the data into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2)

# Shuffle the training data and labels
zipped = list(zip(train_data, train_labels))
random.shuffle(zipped)
train_data, train_labels = zip(*zipped)
train_data = np.array(train_data)
train_labels = np.array(train_labels)

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, train_labels, epochs=10, batch_size=16, validation_data=(val_data, val_labels))

# Plot the accuracy and loss during training
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt
