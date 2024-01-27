import json
import os

import albumentations as al
import cv2
import numpy as np
import torchtext

import torch.utils.data
import torch
import tqdm
from torchtext.data import get_tokenizer
from torchtext.vocab import vocab
from PIL import Image
from torchvision.transforms.v2 import RandomResizedCrop


class TextTokenizerMyCustom(torch.nn.Module):
    def __init__(
            self,
            tokenizer,
            vocabulary,
            max_sentence_length,
            pad_token: str = "<pad>",
            bos_token: str = "<bos>",
            eos_token: str = "<eos>"
    ):
        super().__init__()
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.bos_token = bos_token

        self.max_sentence_length = max_sentence_length
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer

    def forward(self, sentence):
        sentence_splitted: list = self.tokenizer(sentence)
        sentence_splitted.insert(0, self.bos_token)
        sentence_splitted += [self.eos_token]
        sentence_splitted = sentence_splitted[:self.max_sentence_length]
        while len(sentence_splitted) < 20:
            sentence_splitted += [self.pad_token]

        return self.vocabulary(sentence_splitted)


class CocoCaptionsDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            image_size: int,
            images_folder_path,
            captions_json_path,
            text_to_indices
    ):
        self.text_to_indices = text_to_indices
        self.captions_json_path = captions_json_path
        self.images_folder_path = images_folder_path
        self.aug_transform = RandomResizedCrop(
            (image_size, image_size), (0.7, 1)
        )

        with open(captions_json_path, "r") as f:
            self.annotations = json.load(f)["annotations"]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        annotation = self.annotations[item]
        caption = annotation["caption"]
        image_id = annotation["image_id"]
        image_name = f"{image_id:012d}.jpg"

        loaded_image = Image.open(os.path.join(self.images_folder_path, image_name)).convert('RGB')

        loaded_image = self.aug_transform(loaded_image)
        loaded_image = np.divide(loaded_image, 255, dtype=np.float32)[..., ::-1] * 2 - 1
        loaded_image = np.transpose(loaded_image, (2, 0, 1))
        return loaded_image, np.int32(self.text_to_indices(caption))


def build_vocab_from_json(captions_folder_path):
    with open(captions_folder_path, "r") as f:
        loaded_json = json.load(f)
    tokenizer = get_tokenizer("basic_english")

    all_words = []
    sentence_length = []
    for annotation in tqdm.tqdm(loaded_json["annotations"]):
        text = annotation["caption"]
        splitted_text = tokenizer(text)
        all_words += splitted_text
        sentence_length.append(len(splitted_text))

    sentence_length.sort()
    print("Mean sentence length: ", sum(sentence_length) / len(sentence_length))
    print("Max sentence length: ", sentence_length[-1])
    print("Min sentence length: ", sentence_length[0])
    print("Median sentence length: ", sentence_length[len(sentence_length) // 2])

    words_dict = {}
    for i in all_words:
        if i not in words_dict:
            words_dict[i] = 1
        else:
            words_dict[i] += 1

    words_dict = {k: v for k, v in sorted(words_dict.items(), key=lambda item: item[1])}

    v = vocab(
        words_dict,
        min_freq=7,
        specials=["<pad>", "<unk>", "<bos>", "<eos>"]
    )

    v.set_default_index(v["<unk>"])

    return v


def get_dataset(image_size=64, tokenizer_cache_path=None):
    if not (tokenizer_cache_path is not None and os.path.isfile(tokenizer_cache_path)) or tokenizer_cache_path is None:
        text_to_indices = TextTokenizerMyCustom(
            get_tokenizer("basic_english"),
            build_vocab_from_json("/media/valera/SSDM2/ExtractedDatasets/COCO/annotations/captions_train2017.json"),
            20
        )
        if tokenizer_cache_path is not None:
            torch.save(text_to_indices, tokenizer_cache_path)
    else:
        text_to_indices: TextTokenizerMyCustom = torch.load(tokenizer_cache_path)

    dataset = CocoCaptionsDataset(
        image_size,
        "/media/valera/SSDM2/ExtractedDatasets/COCO/train2017",
        "/media/valera/SSDM2/ExtractedDatasets/COCO/annotations/captions_train2017.json",
        text_to_indices
    )
    # print(len(dataset))
    #
    # for _ in range(10000):
    #     image, caption = dataset[np.random.randint(0, len(dataset))]
    #     print(caption)
    #     cv2.imshow("Image", image * 0.5 + 0.5)
    #     cv2.waitKey(0)

    return dataset, text_to_indices


if __name__ == '__main__':
    dataset, _ = get_dataset()
    print(len(dataset))

    for _ in range(10000):
        image, caption = dataset[np.random.randint(0, len(dataset))]
        print(caption)
        cv2.imshow("Image", image.transpose(1, 2, 0) * 0.5 + 0.5)
        cv2.waitKey(0)

