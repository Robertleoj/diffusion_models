# Cell
from birb_dataset import get_bird_dataset
from denoising_diffusion_pytorch import GaussianDiffusion, Unet, Trainer
from birb_dataset import get_stl10_dataset
from birb_dataset import get_animals_10n

# Cell
IMAGE_SIZE = 64
CONDITION_DIM = 256
CONDITIONING_WEIGHTS=(1, 4)
BATCH_SIZE = 16
# ROOT = './data/birds'
ROOT = './data/animals'
RESULTS_FOLDER='./results/animals_cond'

with open(f'{ROOT}/classes.txt', 'r') as f:
    CLASSES = [l.strip() for l in f.readlines()]


# Cell


# Cell
ds, class_to_idx = get_bird_dataset(ROOT, IMAGE_SIZE, CLASSES, CONDITION_DIM)


# ds, class_to_idx = get_stl10_dataset(IMAGE_SIZE)

# Cell
class_to_idx

# Cell
unet = Unet(
    IMAGE_SIZE, 
    condition_dim=CONDITION_DIM,
    condition_vocab_size=len(class_to_idx)
)

diff = GaussianDiffusion(
    unet, 
    image_size=IMAGE_SIZE,
    timesteps=1000,
    conditioning_weights=CONDITIONING_WEIGHTS,
)

trainer = Trainer(
    diff, 
    ds=ds,
    results_folder=RESULTS_FOLDER,
    conditioning=True,
    train_batch_size=BATCH_SIZE,
    class_to_idx=class_to_idx,
    save_and_sample_every=1000,
    train_lr=1e-8,
    num_samples=BATCH_SIZE,
    train_num_steps=10000000000
)

# Cell
trainer.load(101)
trainer.train()


