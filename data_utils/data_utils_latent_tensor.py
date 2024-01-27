import glob

import torch
from diffusers import StableDiffusionPipeline
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torchdata.datapipes.map import SequenceWrapper


def get_dataset():
    latent_paths = glob.glob("/media/valera/SSDM2/LightningFolder/celeba_256x256_latent/*.pt")
    latent_paths.sort()

    dataset = SequenceWrapper(latent_paths)
    dataset = dataset.map(torch.load)

    return dataset


if __name__ == '__main__':
    ds = get_dataset()
    distrib = DiagonalGaussianDistribution(ds[0][None])
    print(distrib.mean.min())

    StableDiffusionPipeline
