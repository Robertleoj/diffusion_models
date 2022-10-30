
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from functools import partial
from torchvision import transforms as T
from pathlib import Path
import os
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ClassEncoding():
    def __init__(self, img_size, channels, classes):
        super().__init__()
        
        self.class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
        self.idx_to_class = sorted(classes)
        
        self.channels = channels
        self.img_size = img_size

        self.embedder = nn.Embedding(len(self.idx_to_class), img_size ** 2)

    def encode(self, klass):

        idx = torch.tensor([self.class_to_idx[klass]], dtype=torch.long)
        emb = self.embedder(idx).view((1, self.img_size, -1))

        return torch.cat(tuple((emb.clone() for _ in range(self.channels))), dim=0)


class ClassCondDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        classes,
        channels=3,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = True,
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.embedder = ClassEncoding(image_size, channels, classes)

        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index, with_embedding=True):
        path = self.paths[index]

        klass = os.path.split(os.path.split(path)[-2])[-1]
        img = Image.open(path)
        img = self.transform(img)

        if with_embedding:
            enc = self.embedder.encode(klass)
            img = torch.cat((enc, img), dim=0)

        return img


def get_bird_dataset(root, img_size):
    with open(f'{root}/classes.txt', 'r') as f:
        classes = [l.strip() for l in f.readlines()]

    bird_ds = ClassCondDataset(
        folder=root, 
        image_size=img_size,
        classes=classes,
        channels=3,
    )

    return bird_ds


