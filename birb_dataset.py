
from mimetypes import init
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
    def __init__(self, cond_dim, class_to_idx):
        super().__init__()
        self.class_to_idx = class_to_idx
        self.embedder = nn.Embedding(len(self.class_to_idx), cond_dim)

        self.embedder.requires_grad_(False)

    def encode(self, klass):

        idx = torch.tensor([self.class_to_idx[klass]], dtype=torch.long)

        return self.embedder(idx)[0]


class ClassCondDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        embedder,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = True,
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size

        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.embedder = embedder
        # self.embedder = ClassEncoding(cond_dim, classes)


        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        klass = os.path.split(os.path.split(path)[-2])[-1]
        img = Image.open(path)
        img = self.transform(img)

        enc = self.embedder.encode(klass)

        return img, enc

def init_bird_encoder(cond_dim, classes):
    class_to_idx = {c: i for i, c in enumerate(sorted(classes))}

    embedder = ClassEncoding(cond_dim, class_to_idx)

    return embedder, class_to_idx

def get_bird_dataset(root, img_size, classes, cond_dim):

    embedder, class_to_idx = init_bird_encoder(cond_dim, classes)

    bird_ds = ClassCondDataset(
        folder=root, 
        image_size=img_size,
        embedder=embedder,
    )

    return bird_ds, embedder, class_to_idx


