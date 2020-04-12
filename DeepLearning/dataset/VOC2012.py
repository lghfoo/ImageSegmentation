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

class VOC2012(VisionDataset):

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

            dataset = VOC2012('./data/VOC2012', split='train')

            img, smnt = dataset[0]
    """

    VOC2012Class = namedtuple('VOC2012Class', ['name', 'id', 'color'])
    #[Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled]
    classes = [
        VOC2012Class('background', 0, (0,0,0)),
        VOC2012Class('aeroplane', 1, (128,0,0)),
        VOC2012Class('bicycle', 2, (0,128,0)),
        VOC2012Class('bird', 3, (128,128,0)),
        VOC2012Class('boat', 4, (0,0,128)),
        VOC2012Class('bottle', 5, (128,0,128)),
        VOC2012Class('bus', 6, (0,128,128)),
        VOC2012Class('car', 7, (128,128,128)),
        VOC2012Class('cat', 8, (64,0,0)),
        VOC2012Class('chair', 9, (192,0,0)),
        VOC2012Class('cow', 10, (64,128,0)),
        VOC2012Class('diningtable', 11, (192,128,0)),
        VOC2012Class('dog', 12, (64,0,128)),
        VOC2012Class('horse', 13, (192,0,128)),
        VOC2012Class('motorbike', 14, (64,128,128)),
        VOC2012Class('person', 15, (192,128,128)),
        VOC2012Class('pottedplant', 16, (0,64,0)),
        VOC2012Class('sheep', 17, (128,64,0)),
        VOC2012Class('sofa', 18, (0,192,0)),
        VOC2012Class('train', 19, (128,192,0)),
        VOC2012Class('tvmonitor', 20, (0,64,128)),
        VOC2012Class('void', 21, (128,64,128))
    ]

    def __init__(self, root, split='train', transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomCrop(size=(64,128)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]), target_transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomCrop(size=(64,128))
        ]), transforms=None):

        super(VOC2012, self).__init__(root, transforms, transform, target_transform)
        self.data_file = os.path.join(self.root, split + '.txt')
        self.split = split
        self.images = []
        self.targets = []
        assert split in ['train', 'test', 'val']
        assert os.path.exists(self.data_file)

        data_file = open(self.data_file, "r")

        lines = data_file.read().splitlines()
        for line in lines:
            splits = line.split(' ')
            img, lab = splits[0], splits[1]
            img_path = os.path.join(self.root, img)
            assert os.path.exists(img_path)
            lab_path = os.path.join(self.root, lab)
            assert os.path.exists(lab_path)
            self.images.append(img_path)
            self.targets.append(lab_path)
        data_file.close()

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


if __name__ == '__main__':
    # test
    root = 'D:\\Study\\毕业设计\\Dataset\\VOC2012Aug\\VOC2012AUG'
    voc2012 = VOC2012(root)
    train_dataloader = torch.utils.data.DataLoader(voc2012, batch_size=4, shuffle=True, num_workers=0)
    for (i,data) in enumerate(train_dataloader):
        print(i)