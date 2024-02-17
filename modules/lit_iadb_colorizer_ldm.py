from typing import Any

import adan_pytorch
import diffusers
import numpy as np
import pytorch_lightning as pl
import torch
from diffusers import AutoencoderKL
from torch.optim import AdamW
from tqdm import trange

import modules.denoising_unet


class ImageToLatent:
    autoencoder: AutoencoderKL = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
    ).eval().requires_grad_(False).float().cuda()

    @staticmethod
    def to_latent(batch):
        with torch.no_grad():
            resulted_latent = ImageToLatent.autoencoder.encode(batch).latent_dist
            resulted_latent.std = 0
            return resulted_latent.sample()


class LitIADBColorizerLDM(pl.LightningModule):
    def __init__(
            self,
            n_sample_timesteps: int = 128,
            n_channels_list=(128, 256, 384, 512),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_blocks_type=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    ):
        super().__init__()
        self.n_sample_timesteps = n_sample_timesteps

        self.inner_model = diffusers.UNet2DModel(
            in_channels=8, out_channels=4,
            down_block_types=down_block_types, block_out_channels=n_channels_list, up_block_types=up_blocks_type
        )

        self.save_hyperparameters()

    def configure_optimizers(self):
        return AdamW(self.parameters(), 1e-4, weight_decay=1e-3)

    def forward_unet(self, *params):
        return self.inner_model(*params, return_dict=False)[0]

    def training_step(self, batch, *args: Any, **kwargs: Any):
        inputs, target_images = batch
        # inputs = ImageToLatent.to_latent(inputs) * 0.18215
        # target_images = ImageToLatent.to_latent(target_images) * 0.18215

        inputs = torch.randn(inputs.size(0), 4, 32, 32, device=inputs.device)
        target_images = torch.randn(target_images.size(0), 4, 32, 32, device=inputs.device)

        alphas = torch.rand(target_images.size(0), device=target_images.device, dtype=target_images.dtype)
        noise = torch.randn_like(target_images)
        alphas_unsqueezed = alphas[:, None, None, None]

        blended = noise * (1 - alphas_unsqueezed) + alphas_unsqueezed * target_images

        conditional_input = torch.cat([inputs, blended], dim=1)

        loss = torch.nn.functional.mse_loss(self.forward_unet(conditional_input, alphas), target_images - noise)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def predict_step(self, batch):
        x_alpha, image_lr = batch

        for i in trange(self.n_sample_timesteps):
            alpha = i / self.n_sample_timesteps
            alpha_next = (i + 1) / self.n_sample_timesteps

            # alpha = 1 - np.cos(alpha * np.pi / 2)
            # alpha_next = 1 - np.cos(alpha_next * np.pi / 2)

            conditional_input = torch.cat([image_lr, x_alpha], dim=1)
            delta_step = self.inner_model(
                conditional_input,
                torch.FloatTensor([alpha]).tile(x_alpha.shape[0]).to("cuda")[:, None]
            )

            x_alpha_next = x_alpha + (alpha_next - alpha) * delta_step

            x_alpha = x_alpha_next

        return x_alpha
