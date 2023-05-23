from os.path import dirname, abspath, join
import sys

PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(join(PROJECT_ROOT, 'src'))

import os
import yadisk

from tqdm.auto import tqdm

from eeg.globals import DATA_DIR


FILES = {
    0: '',  # Links removed due to the dataset license
}

if __name__ == '__main__':
    path = join(DATA_DIR, 'tuh_eeg_h5')
    if not os.path.exists(path):
        os.makedirs(path)

    files = []
    for k, v in FILES.items():
        files.append((k, v))

    d = yadisk.YaDisk()

    for p, link in tqdm(files):
        d.download_public(link, join(path, f'tuh_eeg_{str(p).zfill(3)}.h5'))
