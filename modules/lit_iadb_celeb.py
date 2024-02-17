from typing import Any

import pytorch_lightning as pl
import torch
import diffusers
from tqdm import trange
from ema_pytorch import EMA


class LitIADBCeleb(pl.LightningModule):
    def __init__(
            self,
            image_size: int = 64,
            n_channels_list=(128, 256, 512, 1024),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_blocks_type=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
            n_sample_timesteps: int = 128,
    ):
        super().__init__()
        self.n_channels_list = n_channels_list
        self.n_sample_timesteps = n_sample_timesteps

        self.inner_model = diffusers.models.unet_2d.UNet2DModel(
            image_size,
            in_channels=4,
            out_channels=4,
            down_block_types=down_block_types,
            up_block_types=up_blocks_type,
            block_out_channels=n_channels_list
        )
        self.ema = EMA(
            self.inner_model, include_online_model=False, beta=0.995
        )
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.inner_model.parameters(), 1e-4, weight_decay=5e-3)

    def forward_unet(self, images, timestep):
        return self.inner_model(
            images, timestep
        )[0]

    def forward(self, images, timestep):
        return self.forward_unet(images, timestep)

    def training_step(self, real_images, *args: Any, **kwargs: Any):
        real_images = real_images * 0.18215
        alphas = torch.rand(real_images.size(0), device=real_images.device, dtype=real_images.dtype)
        noise = torch.randn_like(real_images)
        alphas_unsqueezed = alphas[:, None, None, None]

        blended = noise * (1 - alphas_unsqueezed) + alphas_unsqueezed * real_images

        loss = torch.nn.functional.mse_loss(self.forward_unet(blended, alphas), real_images - noise)

        self.log("train_loss", loss, prog_bar=True)
        self.ema.update()

        return loss

    def predict_step(self, batch):
        x_alpha = batch

        for i in trange(self.n_sample_timesteps):
            alpha = i / self.n_sample_timesteps
            alpha_next = (i + 1) / self.n_sample_timesteps
            delta_step = self.ema(
                x_alpha,
                torch.FloatTensor([alpha]).to(self.device)
            ).sample

            x_alpha_next = x_alpha + (alpha_next - alpha) * delta_step

            x_alpha = x_alpha_next

        return x_alpha / 0.18215

