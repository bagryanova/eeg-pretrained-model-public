from os.path import dirname, abspath, join
import sys

PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(join(PROJECT_ROOT, 'src'))

import dn3
import numpy as np
import os

from tqdm.auto import tqdm

from eeg.globals import PREPROCESSED_DATA_DIR, DATA_CONFIG_PATH


if __name__ == '__main__':
    data_config = dn3.configuratron.ExperimentConfig(DATA_CONFIG_PATH)
    dataset = data_config.datasets['sleep-edf']

    mapping = dataset.auto_mapping()

    save_dir = join(PREPROCESSED_DATA_DIR, 'sleep-edf')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for t_id, t in enumerate(tqdm(mapping, unit='person')):
        new_thinker = dataset._construct_thinker_from_config(mapping[t], t)
        new_thinker.add_transform(dn3.transforms.instance.To1020())

        thinker_dir = join(save_dir, str(t_id).zfill(3))
        if not os.path.exists(thinker_dir):
            os.makedirs(thinker_dir)

        for i, (X, y) in enumerate(tqdm(new_thinker)):
            p = join(thinker_dir, f'epoch_{str(i).zfill(4)}')
            np.savez_compressed(p, X=X.numpy(), y=y.numpy())
