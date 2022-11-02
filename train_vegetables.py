from birb_dataset import get_bird_dataset
from denoising_diffusion_pytorch import GaussianDiffusion, Unet, Trainer
from birb_dataset import get_stl10_dataset
from birb_dataset import get_animals_10n

IMAGE_SIZE = 128
CONDITION_DIM = 256
BATCH_SIZE = 8

ROOT = './data/vegetables'
RESULTS_FOLDER='./results/vegetables'

with open(f'{ROOT}/classes.txt', 'r') as f:
    CLASSES = [l.strip() for l in f.readlines()]

ds, class_to_idx = get_bird_dataset(ROOT, IMAGE_SIZE, CLASSES, CONDITION_DIM)

print(class_to_idx)

unet = Unet(
    IMAGE_SIZE, 
    condition_dim=CONDITION_DIM,
    condition_vocab_size=len(class_to_idx)
)

diff = GaussianDiffusion(
    unet, 
    image_size=IMAGE_SIZE,
    timesteps=1000
)

trainer = Trainer(
    diff, 
    ds=ds,
    results_folder=RESULTS_FOLDER,
    conditioning=True,
    train_batch_size=BATCH_SIZE,
    class_to_idx=class_to_idx,
    save_and_sample_every=5000,
    train_lr=1e-6,
    num_samples=32,
    train_num_steps=10000000000
)

trainer.load(4)
trainer.train()


