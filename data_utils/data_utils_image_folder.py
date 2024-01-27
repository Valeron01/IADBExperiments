import glob

import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torchdata


def get_dataset():
    train_dataset = torchdata.datapipes.map.SequenceWrapper(
        glob.glob("")
    )

    train_dataset = torchvision.datasets.CIFAR10(
        "./datasets/cifar10", train=True, transform=transforms.Compose([
            transforms.PILToTensor(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0], [1]),
        ]), download=True
    )

    return train_dataset
