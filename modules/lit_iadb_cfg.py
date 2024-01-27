from typing import Any

import pytorch_lightning as pl
import torch

from modules.optimizers_builder import build_optimizer
from modules.inner_model_factory import build_model


class LitIADBCFG(pl.LightningModule):
    def __init__(
            self,
            inner_model_config,
            optimizer_config,
            n_sample_timesteps: int = 128,
            conditional_scaling: float = 3
    ):
        super().__init__()
        self.conditional_scaling = conditional_scaling
        self.n_sample_timesteps = n_sample_timesteps
        self.inner_model_config = inner_model_config
        self.optimizer_config = optimizer_config

        self.inner_model = build_model(inner_model_config)

        self.save_hyperparameters()

    def configure_optimizers(self):
        return build_optimizer(self.parameters(), self.optimizer_config)

    def training_step(self, batch, *args: Any, **kwargs: Any):
        real_images, labels = batch
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=10)
        labels_onehot = labels_onehot * (torch.rand(labels.size(0), device=labels.device) > 0.1)[:, None]

        alphas = torch.rand(real_images.size(0), device=real_images.device, dtype=real_images.dtype)
        noise = torch.randn_like(real_images)
        alphas_unsqueezed = alphas[:, None, None, None]

        blended = noise * (1 - alphas_unsqueezed) + alphas_unsqueezed * real_images

        conditioning = torch.cat([labels_onehot, alphas[:, None]], dim=1)
        loss = torch.nn.functional.mse_loss(self.inner_model(blended, conditioning), real_images - noise)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def predict_step(self, batch):
        x_alpha, labels = batch
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=10)
        labels_zero = torch.zeros_like(labels_onehot)

        for i in range(self.n_sample_timesteps):
            alpha = i / self.n_sample_timesteps
            alpha_next = (i + 1) / self.n_sample_timesteps
            conditioning = torch.cat([labels_onehot, torch.FloatTensor([alpha]).tile(x_alpha.shape[0]).to(x_alpha)[:, None]], dim=1)

            conditioning_zeros = torch.cat([labels_zero, torch.FloatTensor([alpha]).tile(x_alpha.shape[0]).to(x_alpha)[:, None]], dim=1)
            delta_step_conditioned = self.inner_model(
                x_alpha,
                conditioning
            )

            delta_step_unconditioned = self.inner_model(
                x_alpha,
                conditioning_zeros
            )

            delta_step = (1 + self.conditional_scaling)*delta_step_conditioned - self.conditional_scaling * delta_step_unconditioned

            x_alpha_next = x_alpha + (alpha_next - alpha) * delta_step

            x_alpha = x_alpha_next

        return x_alpha

    def on_train_epoch_end(self) -> None:
        self.eval()

        with torch.no_grad():
            images = self.predict_step((torch.randn(64, 3, 32, 32, device=self.device), torch.randint(0, 10, (64,), device=self.device)))
        self.train()

        self.logger.experiment.add_images("Predicted images", torch.nn.functional.interpolate(images, scale_factor=4), self.global_step)

