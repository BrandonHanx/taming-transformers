import os
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset

class ImagePaths(Dataset):
    def __init__(self, paths):

        self.labels = dict()
        self.labels["file_path_"] = paths
        self._length = len(paths)

        self.rescaler = albumentations.Resize(height=256, width=256)
        self.flip = albumentations.HorizontalFlip(p=0.5)

        self.preprocessor = albumentations.Compose([self.rescaler, self.flip])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class CUHKPEDESTrain(Dataset):
    def __init__(self):
        super().__init__()
        root = "data/cuhkpedes"
        with open("data/cuhkpedes.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class CUHKPEDESVal(Dataset):
    def __init__(self):
        super().__init__()
        root = "data/cuhkpedes"
        with open("data/cuhkpedesval.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
