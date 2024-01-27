import torch.utils.data

import data_utils.data_utils_colorization

import pytorch_lightning as pl

from modules.lit_iadb_colorizer import LitIADBColorizer


def main():
    model = LitIADBColorizer(
        n_features_list=(64, 128, 256, 512, 1024),
        use_attention_list=(False, False, False, False, False)
    )

    trainer = pl.Trainer(
        accelerator="cuda",
        log_every_n_steps=4
    )

    train_dataset = data_utils.data_utils_colorization.get_dataset()
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 7, shuffle=True, num_workers=8, pin_memory=True
    )

    trainer.fit(
        model,
        train_dataloader
    )


if __name__ == '__main__':
    main()
