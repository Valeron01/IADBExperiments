import glob

import numpy as np
import torch.utils.data


import pytorch_lightning as pl

from data_utils.data_utils_sr import RandomImageResizer, ImageSRDataset
from modules.lit_iadbsr import LitIADBSR


def main():
    model = LitIADBSR(
        n_features_list=(64, 128, 128, 256)
    )

    trainer = pl.Trainer(
        accelerator="cuda",
        log_every_n_steps=4
    )

    images_paths = glob.glob(
        "/media/valera/SSDM2/ExtractedDatasets/MineSR/*.*"
    )

    np.random.seed(100)
    np.random.shuffle(images_paths)
    print(images_paths[0])
    np.random.seed()

    print(len(images_paths))

    train_dataloader = torch.utils.data.DataLoader(ImageSRDataset(
        images_paths, RandomImageResizer(144, 144, 4)),
        batch_size=16, shuffle=True, num_workers=8, pin_memory=True
    )

    trainer.fit(
        model,
        train_dataloader
    )


if __name__ == '__main__':
    main()
