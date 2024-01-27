from typing import Any

import pytorch_lightning as pl
import pytorch_lightning.utilities.model_summary
import torch
import diffusers
from tqdm import trange

from modules.optimizers_builder import build_optimizer
from modules.inner_model_factory import build_model
from torch import nn


class LitIADBTextToImage(pl.LightningModule):
    def __init__(
            self,
            vocab_size: int,
            text_dim,
            lstm_hidden_size: int = 1024,
            cross_attention_dim: int = 512,
            image_size: int = 64,
            n_channels_list=(128, 256, 512, 1024),
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_blocks_type=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
            n_sample_timesteps: int = 128,
            conditional_scaling: float = 3,
            **kwargs
    ):
        super().__init__()
        self.n_channels_list = n_channels_list
        self.conditional_scaling = conditional_scaling
        self.n_sample_timesteps = n_sample_timesteps

        self.inner_model = diffusers.models.unet_2d_condition.UNet2DConditionModel(
            image_size,
            in_channels=3,
            out_channels=3,
            down_block_types=down_block_types,
            up_block_types=up_blocks_type,
            block_out_channels=n_channels_list,
            cross_attention_dim=512
        )
        self.null_text_token = nn.Parameter(torch.randn(1, 1, cross_attention_dim))

        # Text processing part
        self.text_embedding = nn.Embedding(
            vocab_size, text_dim
        )

        self.lstm_layer = nn.LSTM(
            text_dim, hidden_size=lstm_hidden_size, num_layers=2, batch_first=True, bidirectional=True
        )
        self.post_stm = nn.Linear(lstm_hidden_size * 2, cross_attention_dim)

        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 2e-4)

    def forward_encode_text(self, text_indices):
        text_embeddings = self.text_embedding(text_indices)
        output, (hn, cn) = self.lstm_layer(text_embeddings)
        output = self.post_stm(output)

        return output

    def forward_unet(self, images, timestep, encoder_states):
        return self.inner_model(
            images, timestep, encoder_states
        )[0]

    def forward(self, images, timestep, text_indices):
        text_embeddings = self.forward_encode_text(text_indices)
        return self.forward_unet(images, timestep, text_embeddings)

    def training_step(self, batch, *args: Any, **kwargs: Any):
        real_images, sentences = batch
        text_features = self.forward_encode_text(sentences)
        null_vector_expanded = torch.tile(self.null_text_token, (sentences.shape[0], sentences.shape[1], 1))
        conditional_pass = torch.rand(real_images.size(0), device=real_images.device) > 0.1
        conditional_pass = conditional_pass[:, None, None].float()

        texts_with_null_labels = text_features * conditional_pass + (1 - conditional_pass) * null_vector_expanded
        alphas = torch.rand(real_images.size(0), device=real_images.device, dtype=real_images.dtype)
        noise = torch.randn_like(real_images)
        alphas_unsqueezed = alphas[:, None, None, None]

        blended = noise * (1 - alphas_unsqueezed) + alphas_unsqueezed * real_images

        loss = torch.nn.functional.mse_loss(self.forward_unet(blended, alphas, texts_with_null_labels), real_images - noise)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def predict_step(self, batch):
        x_alpha, sentences = batch
        with torch.no_grad():
            text_features = self.forward_encode_text(sentences)
            null_vector_expanded = torch.tile(self.null_text_token, (sentences.shape[0], sentences.shape[1], 1))

            for i in trange(self.n_sample_timesteps):
                alpha = i / self.n_sample_timesteps
                alpha_next = (i + 1) / self.n_sample_timesteps

                delta_step_conditioned = self.forward_unet(
                    x_alpha,
                    alpha,
                    text_features
                )

                delta_step_unconditioned = self.forward_unet(
                    x_alpha,
                    alpha,
                    null_vector_expanded
                )

                delta_step = (1 + self.conditional_scaling)*delta_step_conditioned - self.conditional_scaling * delta_step_unconditioned

                x_alpha_next = x_alpha + (alpha_next - alpha) * delta_step

                x_alpha = x_alpha_next

        return x_alpha


if __name__ == '__main__':
    model = LitIADBTextToImage(
        13, 256
    ).cuda()

    print(model(
        torch.randn(8, 3, 64, 64).cuda(),
        torch.rand(8).cuda(),
        torch.randint(0, 13, (8, 64)).cuda()
    ).shape)



