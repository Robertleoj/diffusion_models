
from mimetypes import init
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from functools import partial
from torchvision import transforms as T
from pathlib import Path
from torchvision.datasets import STL10
import os
from PIL import Image, ImageFile
import deeplake

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ClassCondDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        class_to_idx,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = True,
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.class_to_idx = class_to_idx

        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

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
        img = Image.open(path).convert('RGB')
        img = self.transform(img)

        class_idx = self.class_to_idx[klass]

        return img, class_idx



def get_bird_dataset(root, img_size, classes, cond_dim):

    class_to_idx = {c:i for i, c in enumerate(sorted(classes))}

    bird_ds = ClassCondDataset(
        folder=root, 
        image_size=img_size,
        class_to_idx=class_to_idx
    )


    return bird_ds, class_to_idx

def get_stl10_dataset(img_size):
    t = T.Compose([
        T.Resize(img_size),
        T.RandomHorizontalFlip(),
        T.CenterCrop(img_size),
        T.ToTensor()
    ])
    ds = STL10('./data/stl10', download=True, transform=t)

    classes = ds.classes

    class_to_idx = {c:i for i, c in enumerate(classes)}

    return ds, class_to_idx


def get_animals_10n(img_size, batch_size):
    ds = deeplake.load("hub://activeloop/animal10n-train")
    t = T.Compose([
        T.ToPILImage(),
        T.Resize(img_size),
        T.RandomHorizontalFlip(),
        T.CenterCrop(img_size),
        T.ToTensor()
    ])

    dl = ds.pytorch(batch_size=batch_size, transform={
        'images': t, 'labels':None
    }, shuffle=True)

    classes = ds.labels.info

    class_to_idx = {c:i for i, c in enumerate(classes)}

    return ds, dl, class_to_idx
    
    
