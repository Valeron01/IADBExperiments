
import numpy as np
import torch.utils.data
from PIL.Image import open


class RandomImageResizer:
    def __init__(self, height, width, downsample_factor):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.width = width
        self.height = height

    def __call__(self, image):
        image_width = image.width
        image_height = image.height

        max_crop_width = image_height

        crop_width = np.random.randint(self.width, min(image_width, image_height))
        crop_height = round(crop_width / self.width * self.height)

        x_start = np.random.randint(0, image_width - crop_width)
        y_start = np.random.randint(0, image_height - crop_height)

        image_crop = image.crop((x_start, y_start, x_start + crop_width, y_start + crop_height))

        image_hr = image_crop.resize((self.width, self.height))
        image_lr = image_hr.resize((self.width // self.downsample_factor, self.height // self.downsample_factor))

        return image_lr, image_hr



class ImageSRDataset(torch.utils.data.Dataset):
    def __init__(self, images_paths, resizer):
        super().__init__()
        self.resizer = resizer
        self.images_paths = images_paths

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, item):
        image = open(self.images_paths[item])

        lr, hr = self.resizer(image)
        lr = lr.resize(hr.size)

        lr = np.uint8(lr)
        hr = np.uint8(hr)

        lr = torch.from_numpy(lr / 255).float()
        hr = torch.from_numpy(hr / 255).float()

        lr = lr.permute(2, 0, 1)
        hr = hr.permute(2, 0, 1)

        return lr, hr
