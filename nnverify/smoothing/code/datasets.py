from torchvision import transforms, datasets
from typing import *
import torch
import os
import random
from nnverify.common.dataset import Dataset


def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == Dataset.IMAGENET:
        ds = _imagenet(split)
        random.seed(42)
        indices = random.sample(range(len(ds)), 500)
        return [ds[i] for i in indices]
    elif dataset == Dataset.CIFAR10:
        ds = _cifar10(split)
        random.seed(42)
        indices = random.sample(range(len(ds)), 1000)
        return [ds[i] for i in indices]
    elif dataset == Dataset.CIFAR10train:
        dataset == Dataset.CIFAR10
        ds = _cifar10(split)
        return ds
    elif dataset == Dataset.IMAGENET100:
        ds = _imagenet100(split)
        random.seed(42)
        indices = random.sample(range(len(ds)), 500)
        return [ds[i] for i in indices]
    elif dataset == Dataset.IMAGENET10:
        ds = _imagenet10(split)
        random.seed(42)
        indices = random.sample(range(len(ds)), 500)
        return [ds[i] for i in indices]


def get_dataset_full(dataset: str, split: str, data_root) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == Dataset.IMAGENET:
        ds = _imagenet(split, data_root)
    elif dataset == Dataset.CIFAR10:
        ds = _cifar10(split, data_root)
    elif dataset == Dataset.IMAGENET100:
        ds = _imagenet100(split, data_root)
    elif dataset == Dataset.IMAGENET10:
        ds = _imagenet10(split, data_root)
    return ds

def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == Dataset.IMAGENET:
        return 1000
    if dataset == Dataset.IMAGENET100:
        return 100
    if dataset == Dataset.IMAGENET10:
        return 10
    elif dataset == Dataset.CIFAR10:
        return 10


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == Dataset.IMAGENET:
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    if dataset == Dataset.IMAGENET100:
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    if dataset == Dataset.IMAGENET10:
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == Dataset.CIFAR10:
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]


def _cifar10(split: str, data_root) -> Dataset:
    if split == "train":
        return datasets.CIFAR10(data_root, train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.CIFAR10(data_root, train=False, download=True, transform=transforms.ToTensor())


def _imagenet(split: str, data_root) -> Dataset:
    dir = data_root
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    return datasets.ImageFolder(subdir, transform)

def _imagenet100(split: str, data_root) -> Dataset:
    dir = data_root
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    return datasets.ImageFolder(subdir, transform)

def _imagenet10(split: str, data_root) -> Dataset:
    dir = data_root
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    return datasets.ImageFolder(subdir, transform)




class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.means = torch.tensor(means).to(self.device)
        self.sds = torch.tensor(sds).to(self.device)

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).to(input.device)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).to(input.device)
        return (input - means) / sds
