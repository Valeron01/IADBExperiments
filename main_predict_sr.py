import glob

import cv2
import numpy as np
import torch
import torch.utils.data
import torchdata.datapipes.map
from PIL import Image

from modules.lit_iadbsr import LitIADBSR


def image_to_tensor_load(path):
    input_image = Image.open(path)
    input_image = input_image.resize((256, 144)).resize((1024, 576))

    return torch.from_numpy(np.uint8(input_image) / 255).float().permute(2, 0, 1)

input_images = glob.glob("/media/valera/SSDM2/ExtractedDatasets/MinecraftVideo/SimpleSRTest/*.*")
input_images.sort()


# input_images = input_images[]
torch.random.manual_seed(1234484765)
noise = torch.randn(1, 3, 576, 1024).cuda()
model: LitIADBSR = LitIADBSR.load_from_checkpoint(
    glob.glob("/home/valera/PycharmProjects/IADB/lightning_logs/version_36/checkpoints/*.*")[0]
).eval().cuda()

images_dataset = torchdata.datapipes.map.SequenceWrapper(input_images)
images_dataset = images_dataset.map(image_to_tensor_load)
images_loader = torch.utils.data.DataLoader(
    images_dataset, 4, False, num_workers=0, pin_memory=True
)

model.n_sample_timesteps = 64

global_image_index = 0
for input_tensor in images_loader:
    input_tensor = input_tensor.cuda()
    noise_tiled = noise.tile(input_tensor.size(0), 1, 1, 1)
    with torch.no_grad():
        image_sr = model.predict_step((noise_tiled, input_tensor))

    image_up = torch.nn.functional.interpolate(input_tensor, scale_factor=1, mode="bilinear", align_corners=True)
    concated_result = torch.cat([image_up, image_sr], dim=-1).permute(0, 2, 3, 1).cpu().numpy()[..., ::-1]

    for i in concated_result:
        cv2.imwrite(f"/media/valera/SSDM2/LightningFolder/Results/SR4XV2/Frame{global_image_index:06d}.jpg", i * 255)
        global_image_index += 1

