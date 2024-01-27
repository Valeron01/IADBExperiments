import torch


checkpoint = torch.load(
    "/home/valera/PycharmProjects/IADB/lightning_logs/version_8/checkpoints/epoch=8-step=7038.ckpt"
)
print(checkpoint["hyper_parameters"])
