import glob

import cv2
import numpy as np
import pytorch_lightning
import torch
import torchvision.utils
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from tqdm.contrib import tenumerate

from modules.lit_iadb_celeb import LitIADBCeleb


def main():
    model: LitIADBCeleb = LitIADBCeleb.load_from_checkpoint(
        glob.glob("/home/valera/PycharmProjects/IADB/ldm/lightning_logs/version_7/checkpoints/*.*")[0]
    ).eval().requires_grad_(False)
    autoencoder: AutoencoderKL = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
    ).eval().requires_grad_(False).float().cuda()
    model.n_sample_timesteps = 32
    num_images = 100
    print(model.global_step)
    noise = torch.randn(num_images, 4, 32, 32).to(model.device)

    resulted_latent = model.predict_step(noise)
    # resulted_latent = torch.load("/media/valera/SSDM2/LightningFolder/celeba_256x256_latent/025565.pt")[None].cuda()

    for i, latent in tenumerate(resulted_latent):
        resulted_image = autoencoder.decode(latent[None])[0][0].permute(1, 2, 0) * 0.5 + 0.5
        resulted_image = resulted_image.cpu().numpy() * 255
        cv2.imwrite(f"/media/valera/SSDM2/LightningFolder/Results/CelebResults/{i:06d}.jpg", resulted_image)


if __name__ == '__main__':
    main()
