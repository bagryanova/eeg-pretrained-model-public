import numpy as np
import os
import random
import torch
import subprocess

from names_generator import generate_name
from tqdm.auto import tqdm


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class loop_dataloader:
    def __init__(self, dataloader, n_batches):
        self._loader = dataloader
        self._n_batches = n_batches

    def __len__(self):
        return self._n_batches

    def __iter__(self):
        self._step = 0
        self._iter = iter(self._loader)
        return self

    def __next__(self):
        if self._step == self._n_batches:
            raise StopIteration()

        self._step += 1
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self._loader)
            return next(self._iter)


def get_experiment_name(prefix):
    return f'{prefix}_{generate_name()}'


def create_balanced_sampler(dataset, batch_size=64, num_workers=4):
    classes = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    for _, y in tqdm(loader):
        classes.append(y)

    classes = torch.cat(classes)
    _, cnt = np.unique(classes.numpy(), return_counts=True)
    class_weights = 1 / cnt

    weights = torch.tensor([class_weights[s] for s in classes])

    return torch.utils.data.WeightedRandomSampler(weights, len(weights))


def read_file(path):
    with open(path, 'r') as f:
        return f.read()


def generate_name_seed_agnostic():
    return subprocess.check_output(['generate_name']).decode().strip()


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]
