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

    CamVidClass = namedtuple('CamVidClass', ['name', 'id', 'color'])

    classes = [
        CamVidClass('Animal', 0, (64, 128, 64)),
        CamVidClass('Archway', 1, (192, 0, 128)),
        CamVidClass('Bicyclist', 2, (0, 128, 192)),
        CamVidClass('Bridge', 3, (0, 128, 64)),
        CamVidClass('Building', 4, (128, 0, 0)),
        CamVidClass('Car', 5, (64, 0, 128)),
        CamVidClass('CartLuggagePram', 6, (64, 0, 192)),
        CamVidClass('Child', 7, (192, 128, 64)),
        CamVidClass('Column_Pole', 8, (192, 192, 128)),
        CamVidClass('Fence', 9, (64, 64, 128)),
        CamVidClass('LaneMkgsDriv', 10, (128, 0, 192)),
        CamVidClass('LaneMkgsNonDriv', 11, (192, 0, 64)),
        CamVidClass('Misc_Text', 12, (128, 128, 64)),
        CamVidClass('MotorcycleScooter', 13, (192, 0, 192)),
        CamVidClass('OtherMoving', 14, (128, 64, 64)),
        CamVidClass('ParkingBlock', 15, (64, 192, 128)),
        CamVidClass('Pedestrian', 16, (64, 64, 0)),
        CamVidClass('Road', 17, (128, 64, 128)),
        CamVidClass('RoadShoulder', 18, (128, 128, 192)),
        CamVidClass('Sidewalk', 19, (0, 0, 192)),
        CamVidClass('SignSymbol', 20, (192, 128, 128)),
        CamVidClass('Sky', 21, (128, 128, 128)),
        CamVidClass('SUVPickupTruck', 22, (64, 128, 192)),
        CamVidClass('TrafficCone', 23, (0, 0, 64)),
        CamVidClass('TrafficLight', 24, (0, 64, 64)),
        CamVidClass('Train', 25, (192, 64, 128)),
        CamVidClass('Tree', 26, (128, 128, 0)),
        CamVidClass('Truck_Bus', 27, (192, 128, 192)),
        CamVidClass('Tunnel', 28, (64, 0, 64)),
        CamVidClass('VegetationMisc', 29, (192, 192, 0)),
        CamVidClass('Void', 30, (0, 0, 0)),
        CamVidClass('Wall', 31, (64, 192, 0))
    ]

    def __init__(self, root, split='train', transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]), target_transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
        ]), transforms=None):

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

class CamVid11(VisionDataset):

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

    CamVidClass = namedtuple('CamVidClass', ['name', 'id', 'color'])
    #[Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled]
    classes = [
        CamVidClass('Sky', 0, (128,128,128)),
        CamVidClass('Building', 1, (128,0,0)),
        CamVidClass('Column_Pole', 2, (192,192,128)),
        CamVidClass('Road', 3, (128,64,128)),
        CamVidClass('Sidewalk', 4, (60,40,222)),
        CamVidClass('Tree', 5, (128,128,0)),
        CamVidClass('SignSymbol', 6, (192,128,128)),
        CamVidClass('Fence', 7, (64,64,128)),
        CamVidClass('Car', 8, (64,0,128)),
        CamVidClass('Pedestrian', 9, (64,64,0)),
        CamVidClass('Bicyclist', 10, (0,128,192)),
        CamVidClass('Unlabelled', 11, (0,0,0))
    ]

    def __init__(self, root, split='train', transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]), target_transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
        ]), transforms=None):

        super(CamVid11, self).__init__(root, transforms, transform, target_transform)
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