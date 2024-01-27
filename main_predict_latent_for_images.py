import glob

import tqdm
from torchdata.datapipes.map import SequenceWrapper
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.utils
from diffusers.models import AutoencoderKL
import time

from tqdm.contrib import tenumerate


def load_image(path):
    image = cv2.imread(path)
    image = np.divide(image, 255, dtype=np.float32)
    image = image * 2 - 1
    image = torch.from_numpy(image).permute(2, 0, 1)
    return image


def main():
    autoencoder: AutoencoderKL = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
    ).eval().requires_grad_(False).float().cuda()
    print(autoencoder.config)
    images_paths = glob.glob(r"/media/valera/SSDM2/ExtractedDatasets/celeba_hq_256/*.*")
    images_paths.sort()

    images_dataset = SequenceWrapper(images_paths)
    images_dataset = images_dataset.map(load_image)

    images_dataloader = DataLoader(images_dataset, 32, pin_memory=True, num_workers=8)

    predictions = []
    for batch in tqdm.tqdm(images_dataloader):
        resulted_latent = autoencoder.encode(batch.cuda()).latent_dist
        resulted_latent.std = 0
        predictions.append(resulted_latent.sample())
    predictions = torch.cat(predictions, dim=0)
    for i, prediction in tenumerate(predictions):
        torch.save(prediction.cpu(), f"/media/valera/SSDM2/LightningFolder/celeba_256x256_latent/{i:06d}.pt")


if __name__ == '__main__':
    main()
