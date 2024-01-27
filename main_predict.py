import glob

import cv2
import numpy as np
import pytorch_lightning
import torch
import torchvision.utils

from modules.lit_iadb_cfg import LitIADBCFG


def main():
    torch.random.manual_seed(50)
    model: LitIADBCFG = LitIADBCFG.load_from_checkpoint(
        glob.glob("/home/valera/PycharmProjects/IADB/lightning_logs/version_23/checkpoints/*.*")[0]
    ).eval()
    model.conditional_scaling = 2
    model.n_sample_timesteps = 128
    num_images = 100
    noise = torch.randn(num_images, 3, 32, 32).to(model.device)
    classes = torch.randint(0, 10, (num_images,)).to(model.device)

    with torch.no_grad():
        resulted_images = model.predict_step((noise, classes))

    images = torchvision.utils.make_grid(resulted_images, nrow=10)
    images = torch.nn.functional.pad(images, [0, 0, 15, 0])
    images = torch.nn.functional.interpolate(images[None], scale_factor=4)[0].permute(1, 2, 0).cpu().numpy() * 255
    images = np.ascontiguousarray(images)
    images = cv2.putText(images, f"Conditional scale: {model.conditional_scaling}; Iterations: {model.n_sample_timesteps}", (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imwrite(f"/home/valera/PycharmProjects/IADB/resulted_images/Result_cs{model.conditional_scaling}_niters{model.n_sample_timesteps}.png", images[..., ::-1])


if __name__ == '__main__':
    main()
