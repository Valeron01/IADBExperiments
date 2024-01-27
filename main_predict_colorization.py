import glob

import cv2
import numpy as np
import torch
import torch.utils.data
import torchdata.datapipes.map
from PIL import Image

from modules.lit_iadb_colorizer import LitIADBColorizer
from modules.lit_iadbsr import LitIADBSR


def image_to_tensor_load(path):
    input_image = Image.open(path)
    w = input_image.width
    h = input_image.height

    scale_factor = w / 256
    new_h = round(h / scale_factor) // 32 * 32

    input_image = input_image.resize((256, new_h))
    input_image = cv2.cvtColor(np.uint8(input_image), cv2.COLOR_BGR2GRAY)[..., None ]

    return torch.from_numpy(input_image / 255).float().permute(2, 0, 1)


input_images = glob.glob("/media/valera/SSDM2/ExtractedDatasets/COCO/val2017/*.*")
input_images.sort()


# input_images = input_images[]

model: LitIADBColorizer = LitIADBColorizer.load_from_checkpoint(
    glob.glob("/home/valera/PycharmProjects/IADB/lightning_logs/version_43/checkpoints/*.*")[0]
).eval().cuda()

images_dataset = torchdata.datapipes.map.SequenceWrapper(input_images)
images_dataset = images_dataset.map(image_to_tensor_load)
images_loader = torch.utils.data.DataLoader(
    images_dataset, 1, False, num_workers=0, pin_memory=True
)

model.n_sample_timesteps = 64

global_image_index = 0
for input_tensor in images_loader:
    noise = torch.randn(1, 3, input_tensor.shape[2], input_tensor.shape[3]).cuda()
    input_tensor = input_tensor.cuda()
    noise_tiled = noise.tile(input_tensor.size(0), 1, 1, 1)
    with torch.no_grad():
        image_sr = model.predict_step((noise_tiled, input_tensor))

    image_up = input_tensor.tile(1, 3, 1, 1)
    concated_result = torch.cat([image_up, image_sr], dim=-1).permute(0, 2, 3, 1).cpu().numpy()

    for i in concated_result:
        cv2.imwrite(f"/media/valera/SSDM2/LightningFolder/Results/Colorization1/Frame{global_image_index:06d}.jpg", i * 255)
        global_image_index += 1

