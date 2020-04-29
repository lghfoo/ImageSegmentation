import torch
import torchvision
import json
import os
from collections import namedtuple
import zipfile
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import numpy as np
import random

class SiftFlow(VisionDataset):

    """
    Args:
        root (string): Root directory of dataset where directory ``images``
            and ``labels`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val``
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Examples:

        Get semantic segmentation target

        .. code-block:: python

            dataset = SiftFlow('./data/SiftFlow', split='train')

            img, smnt = dataset[0]
    """

    SiftFlowClass = namedtuple('SiftFlowClass', ['name', 'id', 'color'])
    classes = [
        SiftFlowClass('void', 0, (0, 0, 0)),
        SiftFlowClass('awning', 1, (128, 0, 0)),
        SiftFlowClass('balcony', 2, (0, 128, 0)),
        SiftFlowClass('bird', 3, (128, 128, 0)),
        SiftFlowClass('boat', 4, (0, 0, 128)),
        SiftFlowClass('bridge', 5, (128, 0, 128)),
        SiftFlowClass('building', 6, (0, 128, 128)),
        SiftFlowClass('bus', 7, (128, 128, 128)),
        SiftFlowClass('car', 8, (64, 0, 0)),
        # SiftFlowClass('cow', 9, (192, 0, 0)),
        SiftFlowClass('crosswalk', 10, (64, 128, 0)),
        # SiftFlowClass('desert', 11, (192, 128, 0)),
        SiftFlowClass('door', 12, (64, 0, 128)),
        SiftFlowClass('fence', 13, (192, 0, 128)),
        SiftFlowClass('field', 14, (64, 128, 128)),
        SiftFlowClass('grass', 15, (192, 128, 128)),
        # SiftFlowClass('moon', 16, (0, 64, 0)),
        SiftFlowClass('mountain', 17, (128, 64, 0)),
        SiftFlowClass('person', 18, (0, 192, 0)),
        SiftFlowClass('plant', 19, (128, 192, 0)),
        SiftFlowClass('pole', 20, (0, 64, 128)),
        SiftFlowClass('river', 21, (128, 64, 128)),
        SiftFlowClass('road', 22, (0, 192, 128)),
        SiftFlowClass('rock', 23, (128, 192, 128)),
        SiftFlowClass('sand', 24, (64, 64, 0)),
        SiftFlowClass('sea', 25, (192, 64, 0)),
        SiftFlowClass('sidewalk', 26, (64, 192, 0)),
        SiftFlowClass('sign', 27, (192, 192, 0)),
        SiftFlowClass('sky', 28, (64, 64, 128)),
        SiftFlowClass('staircase', 29, (192, 64, 128)),
        SiftFlowClass('streetlight', 30, (64, 192, 128)),
        SiftFlowClass('sun', 31, (192, 192, 128)),
        SiftFlowClass('tree', 32, (0, 0, 64)),
        SiftFlowClass('window', 33, (128, 0, 64)),
    ]

    # ignores = [
    #     9,
    #     11,
    #     16
    # ]

    def __init__(self, root, split='train', transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]), target_transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
        ]), transforms=None):

        super(SiftFlow, self).__init__(root, transforms, transform, target_transform)
        self.images_dir = os.path.join(self.root, 'images', split)
        self.targets_dir = os.path.join(self.root, 'labels', split)
        self.split = split
        self.images = []
        self.targets = []
        assert split in ['train', 'test', 'val']
        assert os.path.exists(self.images_dir)
        assert os.path.exists(self.targets_dir)
        for root, _, files in os.walk(self.images_dir):
            self.images = [os.path.join(root, f) for f in files if os.path.isfile(os.path.join(root, f)) and f.endswith('.jpg')]
            break
        for root, _, files in os.walk(self.targets_dir):
            self.targets = [os.path.join(root, f) for f in files if os.path.isfile(os.path.join(root, f)) and f.endswith('.png')]
            break


    def __getitem__(self, index):
        assert index in range(0, len(self.images))
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])

        # if self.transforms is not None:
        #     image, target = self.transforms(image, target)

        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        if self.transform is not None:
            image = self.transform(image)
        random.seed(seed) # apply this seed to target tranfsorms
        if self.target_transform is not None:
            target = self.target_transform(target)

        target = torch.as_tensor(np.array(target))
        return image, target

    def __len__(self):
        return len(self.images)