import os
import torch
from random import choice
from PIL import Image
from shutil import move


def cerate_testingset(root_dir, source_dirs, class_names, size):
    """Create testing set from traing set."""
    from random import sample

    # If the directory is correctly provided
    if os.path.isdir(os.path.join(root_dir, source_dirs[1])):
        # Creating testing set directories
        os.mkdir(os.path.join(root_dir, 'test'))

        # Iterating through the source directories & creating test directories
        for i, d in enumerate(source_dirs):
            os.rename(os.path.join(root_dir, d),
                      os.path.join(root_dir, class_names[i]))

        for c in class_names:
            os.mkdir(os.path.join(root_dir, 'test', c))

        # Moving selected images
        for c in class_names:
            images = [x for x in os.listdir(os.path.join(
                root_dir, c)) if x.lower().endswith('png')]
            selected_images = sample(images, size)
            for image in selected_images:
                source_path = os.path.join(root_dir, c, image)
                target_path = os.path.join(root_dir, 'test', c, image)
                move(source_path, target_path)
    else:
        print(os.path.join(root_dir, source_dirs[1]))
        raise Exception("Could not find specified directories")


def reorganize(root_dir, class_names):
    if os.path.isdir(os.path.join(root_dir, 'test')):
        for folder in os.listdir(os.path.join(root_dir, 'test')):
            for images in os.listdir(os.path.join(root_dir, 'test', folder)):
                source_path = os.path.join(root_dir, 'test', folder, images)
                target_path = os.path.join(root_dir, folder, images)
                move(source_path, target_path)
            os.rmdir(os.path.join(root_dir, 'test', folder))
        os.rmdir(os.path.join(root_dir, 'test'))
    else:
        print(os.path.join(root_dir, source_dirs[1]))
        raise Exception("Could not find specified directories")


def get_images(class_name, image_dirs):
    # Itaerates through the image directory & loads all the png images
    images = [x for x in os.listdir(
              image_dirs[class_name]) if x.lower().endswith('.png')]
    print(f'Found {len(images)} {class_name}')
    return images


class ChestXRayDataset(torch.utils.data.Dataset):
    """Data Class For Images."""
    def __init__(self, image_dirs, transform, labels):
        """Load the images."""
        self.images = dict()
        self.labels = labels

        for label in self.labels:
            self.images[label] = get_images(label, image_dirs)

        self.image_dirs = image_dirs
        self.transform = transform

    def __len__(self):
        """Return the no. of images in the data."""
        return sum([len(self.images[label]) for label in self.labels])

    def __getitem__(self, index):
        """Return the transform image & label."""
        label = choice(self.labels)
        index %= len(self.images[label])
        image_name = self.images[label][index]
        image_path = os.path.join(self.image_dirs[label], image_name)
        # Converting to RGB according to ResNet classifications
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.labels.index(label)

