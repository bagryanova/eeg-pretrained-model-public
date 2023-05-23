from os.path import dirname, abspath, join
import sys

PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(join(PROJECT_ROOT, 'src'))

import argparse
import h5py
import numpy as np

from mne.io import read_raw_edf
from pathlib import Path
from tqdm.auto import tqdm

from eeg.data.eeg_dataset import EEGDataset
from eeg.globals import EEG_1020_CHANNELS


TARGET_SR = 256


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)

    return parser.parse_args()


def discover_files(dir, extension):
    d = Path(dir)
    return list(map(str, d.rglob('*' + extension)))


def extract_channels(raw):
    channels = []

    for c in EEG_1020_CHANNELS:
        mapped = None
        for raw_c in raw.ch_names:
            if c in raw_c:
                mapped = raw_c
                break

        channels.append(mapped)

    res = []
    for c in channels:
        if c is None:
            res.append(np.zeros((len(raw), 1)))
        else:
            res.append(raw[c][0].reshape(-1, 1))

    return np.concatenate(res, axis=-1)


if __name__ == '__main__':
    args = parse_args()

    files = discover_files(args.input, '.edf')

    print(f'Discovered {len(files)} files.')

    dataset = EEGDataset(args.output, 'w')
    dataset.initialize({
        'eeg': {
            'sample_shape': (19,),
            'chunks': (30 * 256, 19),
            'type': np.float32,
            'compression': None,
        },
        'filename': {
            'sample_shape': tuple(),
            'chunks': True,
            'type': h5py.special_dtype(vlen=str),
            'compression': None,
        },
    })

    for file in tqdm(files):
        raw = read_raw_edf(file, verbose=False, preload=True)
        raw = raw.resample(TARGET_SR)

        dataset.append({
            'eeg': extract_channels(raw),
            'filename': file,
        })

    print(dataset.file['eeg'].attrs['len'] // 256)
