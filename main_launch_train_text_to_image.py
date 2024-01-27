from data_utils.data_utils_coco_captions import get_dataset
from modules.lit_iadb_text_to_image import LitIADBTextToImage
import pytorch_lightning as pl
import torch
import torch.utils.data


def main():
    dataset, tokenizer = get_dataset(64, "./data_utils/TextTokenizerMyCustom.pt")
    dataloder = torch.utils.data.DataLoader(dataset, batch_size=28, shuffle=True, pin_memory=True, num_workers=8)
    trainer = pl.Trainer(
        accelerator="gpu",
        log_every_n_steps=2,
        default_root_dir="./sr"
    )

    model = LitIADBTextToImage(
        len(tokenizer.vocabulary),
        text_dim=512,
        image_size=64,
        n_channels_list=(64, 128, 256, 512),
        additional_params={
            "tokenizer": tokenizer
        }
    )

    trainer.fit(
        model,
        dataloder,
        ckpt_path="/home/valera/PycharmProjects/IADB/sr/lightning_logs/version_17/checkpoints/epoch=0-step=21135.ckpt"
    )


if __name__ == '__main__':
    main()
