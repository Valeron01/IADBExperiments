import torch.utils.data

import data_utils.data_utils_colorization

import pytorch_lightning as pl


from modules.lit_iadb_colorizer_ldm import LitIADBColorizerLDM


def main():
    model = LitIADBColorizerLDM(

    )

    trainer = pl.Trainer(
        accelerator="cuda",
        log_every_n_steps=4
    )

    train_dataset = data_utils.data_utils_colorization.get_dataset()
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 32, shuffle=True, num_workers=8, pin_memory=True
    )

    trainer.fit(
        model,
        train_dataloader
    )


if __name__ == '__main__':
    main()
