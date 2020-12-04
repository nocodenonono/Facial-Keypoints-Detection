from torch.utils.data import Dataset
import torch
import os
from PIL import Image
from scipy.ndimage import rotate
from random import randrange
from torchvision import transforms
import numpy as np
from util import rotate_pt


class MyDataset(Dataset):
    def __init__(self, datadir, images_fp, annotations, transform=None):
        self.datadir = datadir
        self.images = images_fp
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        output: image: PIL image
        """
        image_path = os.path.join(self.datadir, self.images[idx])
        image = Image.open(image_path).convert('L')

        xs = self.annotations[idx][:, 0]
        ys = self.annotations[idx][:, 1]

        annotation = np.append(xs, ys)  # annotation -> np.ndarray: (136,)

        if self.transform:
            image, annotation = self.transform(image, annotation)

        return image, annotation


class ResizeAndRandomRotate(object):
    """
    Randomly rotate the image
    Resize the image if the size is given.

    size should be a tuple
    """

    def __init__(self, degree, size=None):
        if degree <= 0:
            raise RuntimeError('Degree must be positive')

        self.degree = degree
        self.size = size

    def __call__(self, image, annotation):
        """
        :param image: HxWx3 PIL image
        :param annotation: numpy.ndarray -> (136,)
        :return: rotated PIL image, transformed annotations based on rotation.
        """
        x = annotation[:68]
        y = annotation[68:]

        # calculates bounding box boundries based on the landmark pts
        x_min = max(np.amin(x).astype(np.int64) - 30, 0)  # -30 to leave some boundries
        y_min = max(np.amin(y).astype(np.int64) - 30, 0)

        x_max = np.amax(x).astype(np.int64) - x_min + 30
        y_max = np.amax(y).astype(np.int64) - y_min + 30

        # apply bounding box to image
        image = image.crop((x_min, y_min, x_min + x_max, y_min + y_max))

        # remap landmarks based on (x_min, y_min), top left point of bounding box
        x -= x_min
        y -= y_min

        # if no resize, we map to original size
        # else we map to the new resized image
        ox, oy = image.size
        resize = transforms.Resize(self.size)
        new_w, new_h = self.size
        image = resize(image)

        # remap coordinates to (new_w, new_h) coordinates
        x *= new_w / ox
        y *= new_h / oy

        # get random degree
        random_degree = randrange(-self.degree, self.degree)

        # get coordinates of annotation points after rotation
        x, y = rotate_pt(x, y, random_degree, origin=((new_w - 1) / 2, (new_h - 1) / 2))

        # rotate image by degree (counter-clockwise if positive, clockwise otherwise)
        image = rotate(image, angle=random_degree, reshape=False)
        return image, np.append(x, y)


class ToTensor(object):
    def __call__(self, image, annotation):
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        return image, torch.from_numpy(annotation)


class Compose(object):
    """
    This code is obtained in PyTorch Forum
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, annotations):
        for t in self.transforms:
            image, annotations = t(image, annotations)
        return image, annotations
