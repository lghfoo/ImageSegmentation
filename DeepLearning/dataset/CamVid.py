import torch
import torchvision
import json
import os
from collections import namedtuple
import zipfile
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import numpy as np

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

    CamVidClass = namedtuple('CamVidClass', ['name', 'id'])

    classes = [
        CamVidClass('Animal', 0),
        CamVidClass('Archway', 1),
        CamVidClass('Bicyclist', 2),
        CamVidClass('Bridge', 3),
        CamVidClass('Building', 4),
        CamVidClass('Car', 5),
        CamVidClass('CartLuggagePram', 6),
        CamVidClass('Child', 7),
        CamVidClass('Column_Pole', 8),
        CamVidClass('Fence', 9),
        CamVidClass('LaneMkgsDriv', 10),
        CamVidClass('LaneMkgsNonDriv', 11),
        CamVidClass('Misc_Text', 12),
        CamVidClass('MotorcycleScooter', 13),
        CamVidClass('OtherMoving', 14),
        CamVidClass('ParkingBlock', 15),
        CamVidClass('Pedestrian', 16),
        CamVidClass('Road', 17),
        CamVidClass('RoadShoulder', 18),
        CamVidClass('Sidewalk', 19),
        CamVidClass('SignSymbol', 20),
        CamVidClass('Sky', 21),
        CamVidClass('SUVPickupTruck', 22),
        CamVidClass('TrafficCone', 23),
        CamVidClass('TrafficLight', 24),
        CamVidClass('Train', 25),
        CamVidClass('Tree', 26),
        CamVidClass('Truck_Bus', 27),
        CamVidClass('Tunnel', 28),
        CamVidClass('VegetationMisc', 29),
        CamVidClass('Void', 30),
        CamVidClass('Wall', 31),
    ]

    def __init__(self, root, split='train', transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]), target_transform=None, transforms=None):

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
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        target = torch.as_tensor(np.array(target))
        return image, target

    def __len__(self):
        return len(self.images)
