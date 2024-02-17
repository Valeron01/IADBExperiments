import glob
import os
import shutil
import subprocess

import numpy as np
from tqdm.contrib import tzip

latent_paths = glob.glob("/media/valera/SSDM2/ExtractedDatasets/COCO/train2017_latent_128/*.*")
latent_paths.sort()

images_paths = glob.glob("/media/valera/SSDM2/ExtractedDatasets/COCO/train2017/*.*")
images_paths.sort()

for image_path, latent_path in tzip(images_paths, latent_paths):
    target_path = image_path.replace("jpg", "npy").replace("train2017", "train2017_latent_128_renamed")

    # os.makedirs(os.path.dirname(target_path), exist_ok=True)
    # np.load(target_path)

    subprocess.run(["cp", latent_path, target_path])

