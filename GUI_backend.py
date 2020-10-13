import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import torch
import torchvision


ResNetMean = [0.485, 0.456, 0.406]
ResNetSD = [0.229, 0.224, 0.225]
ResNetSize = (227, 227)


mode = None
image = None



def load_model():
    global mode
    path = filedialog.askopenfilename(title="Select Model")
    if mode == 'TF':
        tf.saved_model.load(path)
    else:
        torch.load(path)


def load_image():
    global image
    path = filedialog.askopenfilename(title="Select Image")
    if mode == 'TF':
        image = cv2.imread(path)
        image = cv2.resize(image, (256, 256),
                    interpolation=cv2.INTER_AREA)
    else:
        transform = torchvision.transforms.Compose([
            # According to ResNet classifications
            torchvision.transforms.Resize(size=ResNetSize),
            # Required by PyTorch
            torchvision.transforms.ToTensor(),
            # According to ResNet classifications
            torchvision.transforms.Normalize(mean=ResNetMean, std=ResNetSD)
        ])

        image = [path]
        image = transform(image)
