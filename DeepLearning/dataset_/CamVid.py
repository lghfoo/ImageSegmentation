import torch
import torchvision
import json
import os
from collections import namedtuple
import zipfile
from torchvision.datasets.vision import VisionDataset

from PIL import Image

torchvision.datasets.Cityscapes

class CamVid(VisionDataset):
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

            dataset = CamVid('./data/camvid', split='train')

            img, smnt = dataset[0]
    """
    def __init__(self, root, split='train', transform=None, target_transform=None, transforms=None):
        super(CamVid, self).__init__(root, transforms, transform, target_transform)
        self.images_dir = os.path.join(self.root, 'images', split)
        self.targets_dir = os.path.join(self.root, 'labels', split)
        self.split = split
        self.images = []
        self.targets = []
        assert split in ['train', 'test', 'val']
        assert os.path.exists(self.images_dir)
        assert os.path.exists(self.targets_dir)
        for root, _, files in os.walk(self.images_dir):
            self.images = [os.path.join(root, f) for f in files if os.path.isfile(os.path.join(root, f)) and f.endswith('.png')]
            break
        for root, _, files in os.walk(self.targets_dir):
            self.targets = [os.path.join(root, f) for f in files if os.path.isfile(os.path.join(root, f)) and f.endswith('.png')]
            break


    def __getitem__(self, index):
        assert index in range(0, len(self.images))
        image = Image.open(self.images[index]).convert('RGB')
        target =Image.open(self.targets[index]).convert('RGB')
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self):
        return len(self.images)
