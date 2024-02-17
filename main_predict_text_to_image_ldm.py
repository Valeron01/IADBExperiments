import glob
import math

import cv2
import torch
import torchvision.utils
from diffusers import AutoencoderKL

from modules.lit_iadb_text_to_image_ldm import LitIADBTextToImageLDM


device = "cuda"

model: LitIADBTextToImageLDM = LitIADBTextToImageLDM.load_from_checkpoint(
    glob.glob("/home/valera/PycharmProjects/IADB/text_to_image/checkpoints/31/*.ckpt")[0],
    map_location=device
).eval().to(device)
print("Train steps passed: ", model.global_step)
model.conditional_scaling = 3
model.n_sample_timesteps = 128
tokenizer = model.hparams["additional_params"]["tokenizer"]
# a beautiful evening landscape with river mountains and deep fog
input_sentence = "a beautiful sunrise on a beach"

n_images = 9
input_tensor = torch.IntTensor(tokenizer(input_sentence)).to(device)
print(input_tensor)
latent = model.predict_step(
    (
        torch.randn(n_images, 4, 32, 32).to(device),
        input_tensor[None].tile(n_images, 1)
    )
)
autoencoder: AutoencoderKL = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
).eval().requires_grad_(False).half().to(device)

resulted_image = autoencoder.decode(latent.half(), return_dict=False)[0]

resulted_image = torchvision.utils.make_grid(resulted_image * 127.5 + 127.5, nrow=round(math.sqrt(n_images)))

cv2.imwrite(
    "./resulted_image_ldm_cosine.png",
    resulted_image.permute(1, 2, 0).cpu().numpy()
)

