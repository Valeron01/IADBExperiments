import glob
import math

import tqdm
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torchdata.datapipes.map import SequenceWrapper
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.utils
from diffusers.models import AutoencoderKL
import time

from tqdm.contrib import tenumerate


def crop_centered(image):
    height, width, _ = image.shape

    if height > width:
        crop_size = (height - width) / 2
        return image[math.floor(crop_size):math.floor(-crop_size), :]
    elif width > height:
        crop_size = (width - height) / 2
        return image[:, math.floor(crop_size):math.floor(-crop_size)]
    return image


def load_image(path):
    image = cv2.imread(path)
    image = crop_centered(image)
    image = cv2.resize(image, (128, 128))
    image = np.divide(image, 255, dtype=np.float32)
    image = image * 2 - 1
    image = torch.from_numpy(image).permute(2, 0, 1)
    return image


def main():
    autoencoder: AutoencoderKL = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
    ).eval().requires_grad_(False).float().cuda()

    # autoencoder = torch.compile(autoencoder, mode="reduce-overhead")

    print(autoencoder.config)
    images_paths = glob.glob(r"/media/valera/SSDM2/ExtractedDatasets/COCO/train2017/*.*")
    images_paths.sort()

    images_dataset = SequenceWrapper(images_paths)
    images_dataset = images_dataset.map(load_image)

    images_dataloader = DataLoader(images_dataset, 32, pin_memory=True, num_workers=8)

    image_index = 0
    for batch in tqdm.tqdm(images_dataloader):
        resulted_latent = autoencoder.encode(batch.cuda()).latent_dist
        resulted_latent.std = 0
        resulted_latent = resulted_latent.sample()

        for latent in resulted_latent.cpu():
            latent = latent.numpy()
            np.save(f"/media/valera/SSDM2/ExtractedDatasets/COCO/train2017_latent_128/{image_index:06d}", latent)
            image_index += 1


if __name__ == '__main__':
    main()
