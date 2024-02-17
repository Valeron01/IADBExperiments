from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data_utils.data_utils_coco_captions_ldm import get_dataset
from modules.lit_iadb_text_to_image_ldm import LitIADBTextToImageLDM
import pytorch_lightning as pl
import torch
import torch.utils.data


def main():
    dataset, tokenizer = get_dataset("./data_utils/TextTokenizerMyCustom.pt")
    dataloder = torch.utils.data.DataLoader(dataset, batch_size=192, shuffle=True, pin_memory=True, num_workers=8)
    logger = TensorBoardLogger("./text_to_image")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"/home/valera/PycharmProjects/IADB/text_to_image/checkpoints/{logger.version}"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        log_every_n_steps=2,
        default_root_dir="./text_to_image",
        accumulate_grad_batches=1,
        callbacks=[checkpoint_callback]
    )

    model = LitIADBTextToImageLDM(
        len(tokenizer.vocabulary),
        text_dim=512,
        image_size=16,
        n_channels_list=(128, 384, 512, 768),
        additional_params={
            "tokenizer": tokenizer
        }
    )

    trainer.fit(
        model,
        dataloder,
        # ckpt_path="/home/valera/PycharmProjects/IADB/text_to_image/checkpoints/30/epoch=18-step=87856.ckpt"
    )


if __name__ == '__main__':
    main()
