from torch.utils.data import Dataset
import glob
import json
import os
import albumentations as al
import cv2
import numpy as np
from tqdm import tqdm


class ImagesColorizerDataset(Dataset):
    def __init__(self, images_paths):
        self.images_paths = images_paths
        self.images_transform = al.Sequential([
            al.RandomResizedCrop(192, 192, scale=(0.08, 1)),
        ], p=1)

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, item):
        image = cv2.imread(self.images_paths[item])
        image = np.divide(image, 255, dtype=np.float32)
        image = self.images_transform(image=image)["image"]
        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[..., None]

        return image_bw.transpose(2, 0, 1), image.transpose(2, 0, 1)


def get_dataset():
    with open("/media/valera/SSDM2/ExtractedDatasets/Imagenet/test_large_colored.json", "r") as f:
        images_paths = json.load(f)
    print("Dataset size is: ", len(images_paths))
    dataset = ImagesColorizerDataset(images_paths)
    return dataset


def is_bw(image):
    b = image[..., 0]
    g = image[..., 1]
    r = image[..., 2]

    return (r == g).all()


if __name__ == '__main__':
    dataset = get_dataset()
    for bw, color in dataset:
        bw = np.tile(bw, [3, 1, 1])
        concated = np.concatenate([bw, color], axis=2)
        cv2.imshow("cghhg", concated.transpose(1, 2, 0))
        cv2.waitKey(0)