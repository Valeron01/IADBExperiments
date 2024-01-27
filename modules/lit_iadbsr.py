from typing import Any

import adan_pytorch
import numpy as np
import pytorch_lightning as pl
import torch
from tqdm import trange

import modules.denoising_unet

class LitIADBSR(pl.LightningModule):
    def __init__(
            self,
            n_sample_timesteps: int = 128,
            n_features_list=(32, 64, 128, 128),
            use_attention_list=(False, False, False, False)
    ):
        super().__init__()
        self.n_sample_timesteps = n_sample_timesteps

        self.inner_model = modules.denoising_unet.DenoisingUNet(
            in_embed_dim=1,
            in_channels=6,
            n_features_list=n_features_list,
            use_attention_list=use_attention_list
        )

        self.save_hyperparameters()

    def configure_optimizers(self):
        return adan_pytorch.Adan(self.parameters(), 1e-4)

    def training_step(self, batch, *args: Any, **kwargs: Any):
        inputs, target_images = batch

        alphas = torch.rand(target_images.size(0), device=target_images.device, dtype=target_images.dtype)
        noise = torch.randn_like(target_images)
        alphas_unsqueezed = alphas[:, None, None, None]

        blended = noise * (1 - alphas_unsqueezed) + alphas_unsqueezed * target_images

        conditional_input = torch.cat([inputs, blended], dim=1)

        loss = torch.nn.functional.mse_loss(self.inner_model(conditional_input, alphas[:, None]), target_images - noise)

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
                torch.FloatTensor([alpha]).tile(x_alpha.shape[0]).to(x_alpha)[:, None]
            )

            x_alpha_next = x_alpha + (alpha_next - alpha) * delta_step

            x_alpha = x_alpha_next

        return x_alpha
