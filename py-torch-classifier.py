"""
Chest X-Ray Covid-19 Classifier

Based on https://www.coursera.org/projects/covid-19-detection-x-ray
Dataset from https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
"""
# Importing Modules
import random
import torch
import torchvision

from sys import argv
import numpy as np
from termcolor import cprint
from helper import *
from PIL import Image
from matplotlib import pyplot as plt

# Pre-Defining lists used later
class_names = ['normal', 'viral', 'covid']
ResNetMean = [0.485, 0.456, 0.406]
ResNetSD = [0.229, 0.224, 0.225]
ResNetSize = (227, 227)
batch_size = 6
TestingSize = 30


# DEFINED IN helper.py
cerate_testingset('Data',
                  ['normal', 'viral', 'covid'],
                  class_names, TestingSize)


# Training Transformation Object
train_transform = torchvision.transforms.Compose([
    # According to ResNet classifications
    torchvision.transforms.Resize(size=ResNetSize),
    # A little bit of data augmentation to allow for inconsistencies.
    torchvision.transforms.RandomHorizontalFlip(0.2),
    torchvision.transforms.RandomVerticalFlip(0.2),
    # Required by PyTorch
    torchvision.transforms.ToTensor(),
    # According to ResNet classifications
    torchvision.transforms.Normalize(mean=ResNetMean, std=ResNetSD)
])

# Testing Transformation Object
test_transform = torchvision.transforms.Compose([
    # According to ResNet classifications
    torchvision.transforms.Resize(size=ResNetSize),
    # Required by PyTorch
    torchvision.transforms.ToTensor(),
    # According to ResNet classifications
    torchvision.transforms.Normalize(mean=ResNetMean, std=ResNetSD)
])

# Directory Locations
train_dirs = {
    'normal': 'Data/normal',
    'viral': 'Data/viral',
    'covid': 'Data/covid'
}
test_dirs = {
    'normal': 'Data/test/normal',
    'viral': 'Data/test/viral',
    'covid': 'Data/test/covid'
}

train_dataset = ChestXRayDataset(train_dirs, train_transform, class_names)
test_dataset = ChestXRayDataset(test_dirs, test_transform, class_names)

dl_train = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
dl_test = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True)

cprint(f'Number of training batches {len(dl_train)}', 'yellow')
cprint(f'Number of test batches {len(dl_test)}', 'yellow')


def show_images(images, labels, predictions):
    plt.figure(figsize=(8, 4))
    for i, image in enumerate(images):
        plt.subplot(1, 6, i + 1, xticks=[], yticks=[])
        image = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0., 1.)
        plt.imshow(image)
        col = 'green'
        if predictions[i] != labels[i]:
            col = 'red'

        plt.xlabel(f'{class_names[int(labels[i].numpy())]}')
        plt.ylabel(f'{class_names[int(predictions[i].numpy())]}', color=col)
    plt.tight_layout()
    plt.show()

# Downloading the model & setting parameters
resnet152 = torchvision.models.resnet152(pretrained=True)
resnet152.fc = torch.nn.Linear(2048, 6, True)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(resnet152.parameters(), lr=3e-5)

# Shows images according to predictions
def show_predictions():
    """Display images according to model prections."""
    resnet152.eval()
    images, labels = next(iter(dl_test))
    outputs = resnet152(images)
    index, predictions = torch.max(outputs, 1)
    show_images(images, labels, predictions)


def train(epochs, target):
    cprint("Starting Training..." , 'green')
    for epoch in range(0, epochs):
        cprint(f'Starting epoch {epoch+1}/{epochs}'.center(os.get_terminal_size().columns - 5, "="), 'green')

        train_loss = 0
        # Set model to training mode
        resnet152.train()

        for train_step, (images, labels) in enumerate(dl_train):
            # Reset gradient descent to zero
            optimizer.zero_grad()
            # Getting predictions
            predictions = resnet152(images)
            # Evaluating loss
            loss = loss_fn(predictions, labels)
            # Backpropgate to update weights
            loss.backward()
            # Complete a gradient step
            optimizer.step()
            # Update training loss
            train_loss += loss.item()

            # Every 20 iterations, evaluae the model
            if train_step % 20 == 0:
                cprint(f"Evaluating at iteartion {train_step}", 'green')
                val_loss = 0
                acc = 0
                # Set model to evaluation mode
                resnet152.eval()

                for val_step, (images, labels) in enumerate(dl_test):
                    predictions = resnet152(images)
                    loss = loss_fn(predictions, labels)
                    val_loss += loss.item()
                    index, predictions = torch.max(predictions, 1)
                    acc += sum((predictions == labels).numpy())
                val_loss /= (val_step + 1)
                acc /= len(test_dataset)
                cprint(
                    f'Validation Loss: {val_loss:.4f} || Validation Accuracy: {acc*100:.4f}%', attrs=['bold'])
                # show_predictions()
                resnet152.train()

                if acc > target:
                    cprint(f"Target Reached. Current Accuracy: {(acc*100)}%", 'green', attrs=['bold'])
                    return

        train_loss /= (train_step + 1)
        cprint(f'Training Loss: {train_loss:.4f}', 'yellow')
    cprint('Training completed but target not reached', 'green', attrs=['bold'])

train(3, 0.95)
show_predictions()

reorganize('Data',
           ['normal', 'viral', 'covid'])

if len(argv) == 2:
    torch.save(resnet152, argv[1])
