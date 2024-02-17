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
        n_channels_list=(224, 448, 448, 672)
    )

    trainer.fit(
        model,
        dataloder,
        ckpt_path="/home/valera/PycharmProjects/IADB/ldm/lightning_logs/version_13/checkpoints/epoch=34-step=16415.ckpt"
    )


if __name__ == '__main__':
    main()
