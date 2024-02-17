import math

import cv2
import torch
import torchvision.utils

from modules.lit_iadb_text_to_image import LitIADBTextToImage

model: LitIADBTextToImage = LitIADBTextToImage.load_from_checkpoint(
    "/media/valera/SSDM2/ImagesGenerator/epoch=20-step=443835.ckpt",
    map_location="cuda"
).eval().cuda()
model.conditional_scaling = 3
tokenizer = model.hparams["additional_params"]["tokenizer"]

input_sentence = "A vase with green leaves is sitting on display."

n_images = 4
input_tensor = torch.IntTensor(tokenizer(input_sentence)).cuda()
print(input_tensor)

resulted_image = model.predict_step(
    (
        torch.randn(n_images, 3, 64, 64).cuda(),
        input_tensor[None].tile(n_images, 1)
    )
)

resulted_image = torchvision.utils.make_grid(resulted_image * 127.5 + 127.5, nrow=round(math.sqrt(n_images)))

cv2.imwrite(
    "./resulted_image.png",
    resulted_image.permute(1, 2, 0).cpu().numpy()
)

