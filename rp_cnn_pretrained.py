import os
import cv2
import numpy as np
import tensorflow as tf
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPool2D,
    Flatten,
    Dense,
    Activation,
    BatchNormalization,
    GlobalAveragePooling2D,
)
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import glob
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter("ignore", FutureWarning)

# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# print("Num GPUs Available: ", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

n_train_normal = 437
n_train_stroke = 438
n_valid_normal = 93
n_valid_stroke = 99
n_test_normal = 48
n_test_stroke = 48

# Organize data into train, valid, test dirs
os.chdir("rp_data")
if os.path.isdir("train/normal") is False:
    os.makedirs("train/normal")
    os.makedirs("train/stroke")
    os.makedirs("valid/normal")
    os.makedirs("valid/stroke")
    os.makedirs("test/normal")
    os.makedirs("test/stroke")

    for c in random.sample(glob.glob("*normal*"), n_train_normal):
        shutil.move(c, "train/normal")
    for c in random.sample(glob.glob("*weak*"), n_train_stroke):
        shutil.move(c, "train/stroke")
    for c in random.sample(glob.glob("*normal*"), n_valid_normal):
        shutil.move(c, "valid/normal")
    for c in random.sample(glob.glob("*weak*"), n_valid_stroke):
        shutil.move(c, "valid/stroke")
    for c in random.sample(glob.glob("*normal*"), n_test_normal):
        shutil.move(c, "test/normal")
    for c in random.sample(glob.glob("*weak*"), n_test_stroke):
        shutil.move(c, "test/stroke")

os.chdir("../")

train_path = "rp_data/train"
valid_path = "rp_data/valid"
test_path = "rp_data/test"

# Define image size and number of channels
img_height = 465
img_width = 930

train_batches = ImageDataGenerator(
    tf.keras.applications.vgg16.preprocess_input
).flow_from_directory(
    directory=train_path,
    target_size=(img_height, img_width),
    classes=["normal", "stroke"],
    batch_size=10,
)
valid_batches = ImageDataGenerator(
    tf.keras.applications.vgg16.preprocess_input
).flow_from_directory(
    directory=valid_path,
    target_size=(img_height, img_width),
    classes=["normal", "stroke"],
    batch_size=10,
)
test_batches = ImageDataGenerator(
    tf.keras.applications.vgg16.preprocess_input
).flow_from_directory(
    directory=test_path,
    target_size=(img_height, img_width),
    classes=["normal", "stroke"],
    batch_size=10,
    shuffle=False,
)

# Make sure tghe batches have the correct shape
assert train_batches.n == n_train_normal + n_train_stroke
assert valid_batches.n == n_valid_normal + n_valid_stroke
assert test_batches.n == n_test_normal + n_test_stroke
assert (
    train_batches.num_classes
    == valid_batches.num_classes
    == test_batches.num_classes
    == 2
)


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


vgg16_model = tf.keras.applications.vgg16.VGG16()
vgg16_model.summary()

model = Sequential()
for layer in vgg16_model.layers[:-4]:
    model.add(layer)

for layer in model.layers:
    layer.trainable = False

model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(units=2, activation="softmax"))

model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)

test_imgs, test_labels = next(test_batches)
# plotImages(test_imgs)
# print(test_labels)

# test_batches.classes

predictions = model.predict(x=test_batches, verbose=0)

np.round(predictions)

cm = confusion_matrix(
    y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1)
)


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


print(test_batches.class_indices)

score = model.evaluate(test_batches, verbose=0)
print("Test loss: {}, Test accuracy {}".format(score[0], score[1]))

cm_plot_labels = ["normal", "stroke"]
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title="Confusion Matrix")
