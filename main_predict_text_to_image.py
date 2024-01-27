import cv2
import torch

from modules.lit_iadb_text_to_image import LitIADBTextToImage

model: LitIADBTextToImage = LitIADBTextToImage.load_from_checkpoint(
    "./sr/lightning_logs/version_18/checkpoints/epoch=20-step=443835.ckpt",
    map_location="cuda"
).eval().cuda()
model.conditional_scaling = 3
tokenizer = model.hparams["additional_params"]["tokenizer"]

input_sentence = "A group of boys playing baseball"
input_tensor = torch.IntTensor(tokenizer(input_sentence)).cuda()
print(input_tensor)

resulted_image = model.predict_step(
    (
        torch.randn(1, 3, 64, 64).cuda(),
        input_tensor[None].cuda()
    )
)

cv2.imwrite(
    "./resulted_image.png",
    resulted_image[0].permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5
)

