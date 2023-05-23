from os.path import dirname, abspath, join
import sys

PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(join(PROJECT_ROOT, 'src'))

import numpy as np

from eeg.data.eeg_dataset import EEGDataset, EpochedDataset, EpochedClassificationDataset
from eeg.globals import DATA_DIR


class SequentialEpochedClassificationDataset(EpochedDataset):
    def __init__(self, eeg_dataset, sequence_length, sub_indices=None)
        self._dataset = eeg_dataset
        self._dataset.open()
        self._thinkers = np.zeros(self._dataset.file['label'].attrs['len'], dtype=int)

        for i in range(self._dataset.file['thinker'].attrs['len']):
            ind = self._dataset.file['_ind'][i]
            self._thinkers[ind[0]:ind[1]] = self._dataset.file['thinker'][i]

        if sub_indices is None:
            self._sub_indices = np.arange(self._thinkers.shape[0], dtype=int)
        else:
            self._sub_indices = sub_indices

        self._sequence_length = sequence_length

    def __getitem__(self, idx):
        self._dataset.open()
        X = self._dataset.file['eeg'][self._sub_indices[idx]].T
        y = self._dataset.file['label'][self._sub_indices[idx]]
        return X, y

    def _subset(self, ind):
        return SequentialEpochedClassificationDataset(
            self._dataset,
            self._sequence_length,
            ind,
        )


if __name__ == '__main__':
    dataset = EpochedClassificationDataset(
        EEGDataset(join(DATA_DIR, 'sleep-edfx-1.0.0-30min-wake-labels.h5')),
    )

    for train, val, test in dataset.lmso(folds=5, shuffle=True):
        print(len(train), len(val), len(test))
        print(train.thinkers)
        print(val.thinkers)
        print(test.thinkers)
        print()
