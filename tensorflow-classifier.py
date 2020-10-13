"""
Tesorflow Model Trainer for image classification.
Dataset from https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
"""
import cv2
import numpy as np
import tensorflow as tf
from termcolor import cprint
from sys import argv, exit
import os
from sklearn.model_selection import train_test_split


# Global Variables
EPOCHS = 10
IMG_WIDTH = 128
IMG_HEIGHT = 128
NUM_CATEGORIES = 3
TEST_SIZE = 0.25
Weights = None
ResNet = False
PreTrained = False


# Numeric Values for Labels
numeric = {
    "covid": 1,
    "normal": 0,
    "viral": 2
}


def main():
    """Trains & Evaluates a model for Image Classification."""
    global Weights, ResNet, PreTrained
    # Check command-line arguments
    if len(argv) not in [2, 3, 4]:
        cprint("Usage: python tensorflow-classifier.py data_directory\
               [{optional} resnet] [{optional} model.h5]", 'red',
               attrs=['bold'])
        exit(1)

    if len(argv) == 3:
        if argv[2].upper() == 'resnetpt':
            ResNet = True
            PreTrained = True
        if argv[2].upper() == 'resnet':
            ResNet = True

    # Get image arrays and labels for all image files
    images, labels = load_data(argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Speeds up things by using Just In Time Execution
    tf.keras.backend.clear_session()
    tf.config.optimizer.set_jit(True)

    # Get a compiled neural network
    model = get_model(resnet=ResNet, pretrained=PreTrained)

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(argv) == 4:
        filename = argv[3]
        model.save(filename)
        cprint(f"Model saved to {filename}.", 'green', attrs=['bold'])


def load_data(data_dir):
    """Load images from directory `data_dir`."""
    # Initializing 2 empty lists
    images = []
    labels = []

    # Iterating through all the folders in the directory
    for folder in os.listdir(data_dir):
        # Iterating through all the images in the folder
        for image_path in os.listdir(os.path.join(data_dir, folder)):
            # Reading the image
            image = cv2.imread(os.path.join(data_dir, folder, image_path),
                               cv2.IMREAD_COLOR)
            # Resizing the image
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT),
                               interpolation=cv2.INTER_AREA)
            # Adding image to images list
            images.append(image)
            # Adding corresponding label to the label list (i.e. folder name)
            labels.append(int(numeric[folder]))

    # Return the images & labels list
    return images, labels


def get_model(resnet=False, pretrained=False):
    """Return a compiled convolutional neural network model."""
    if resnet:
        if pretrained:
            Weights = 'imagenet'
        model = tf.keras.applications.ResNet152(weights=Weights,
                                                input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
                                                pooling='max', classes=3)

        # Training the neural network
        model.compile(
            optimizer="adamax",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Return the model
        return model

    model = tf.keras.models.Sequential([
        # # Data Augmentation Layer [Requires tensorflow nightly]
        # tf.keras.layers.experimental.preprocessing.RandomFlip(
        #     "horizontal_and_vertical"),
        # tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),

        # Convolutional Layer with 12 fiters, 3*3 kernel, relu activation
        tf.keras.layers.SeparableConv2D(
            22, (3, 3), activation="elu",
            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        # Convolutional Layer with 12 fiters, 3*3 kernel, relu activation
        tf.keras.layers.Conv2D(
            12, (3, 3), activation="relu",
            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # Average Spatial Pooling using 3*3 matrix
        tf.keras.layers.AveragePooling2D(3),
        # Flatten units
        tf.keras.layers.Flatten(),
        # Add 2 hidden layers with dropouts in each layer
        tf.keras.layers.Dense(45, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(45, activation="relu"),
        tf.keras.layers.Dropout(0.1),

        # Final layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Training the neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"]
    )

    # Return the model
    return model


if __name__ == "__main__":
    main()
