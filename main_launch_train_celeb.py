from data_utils.data_utils_latent_tensor import get_dataset
from modules.lit_iadb_celeb import LitIADBCeleb
import pytorch_lightning as pl
import torch
import torch.utils.data


def main():
    dataset = get_dataset()
    dataloder = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=8)
    trainer = pl.Trainer(
        accelerator="gpu",
        log_every_n_steps=2,
        default_root_dir="./ldm",
    )

    model = LitIADBCeleb(
        n_channels_list=(128, 256, 384, 512)
    )

    trainer.fit(
        model,
        dataloder,
    )


if __name__ == '__main__':
    main()
