from data_utils.data_utils_cifar10 import get_dataset
from modules.lit_iadb_cfg import LitIADBCFG
import pytorch_lightning as pl
import torch
import torch.utils.data


def main():
    dataset = get_dataset()
    dataloder = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=8)

    model_config = {
        "module_name": "DenoisingUNet",
        "n_features_list": (64, 128, 256, 512),
        "use_attention_list": (False, True, True, True)
    }
    optimizer_config = {
        "name": "Adan",
        "lr": 1e-4
    }

    trainer = pl.Trainer(
        accelerator="gpu",
        log_every_n_steps=2,
        default_root_dir="./sr"
    )

    model = LitIADBCFG(model_config, optimizer_config)

    trainer.fit(model, dataloder)


if __name__ == '__main__':
    main()
