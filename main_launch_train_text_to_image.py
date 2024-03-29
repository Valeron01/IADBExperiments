from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data_utils.data_utils_coco_captions import get_dataset
from modules.lit_iadb_text_to_image import LitIADBTextToImage
import pytorch_lightning as pl
import torch
import torch.utils.data


def main():
    dataset, tokenizer = get_dataset(32, "./data_utils/TextTokenizerMyCustom.pt")
    dataloder = torch.utils.data.DataLoader(dataset, batch_size=36, shuffle=True, pin_memory=True, num_workers=8)
    logger = TensorBoardLogger("./text_to_image")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"/home/valera/PycharmProjects/IADB/text_to_image/checkpoints/{logger.version}",
        every_n_train_steps=3000
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        log_every_n_steps=2,
        default_root_dir="./text_to_image",
        accumulate_grad_batches=4,
        callbacks=[checkpoint_callback]
    )

    model = LitIADBTextToImage(
        len(tokenizer.vocabulary),
        text_dim=512,
        image_size=32,
        n_channels_list=(128, 384, 512, 768),
        additional_params={
            "tokenizer": tokenizer
        }
    )

    trainer.fit(
        model,
        dataloder
    )


if __name__ == '__main__':
    main()
