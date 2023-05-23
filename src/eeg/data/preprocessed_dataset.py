import os
import torch
import numpy as np


class PreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, path, thinkers=None):
        self._path = path

        self._thinkers = []
        for thinker in os.listdir(path) if thinkers is None else thinkers:
            if not os.path.isdir(os.path.join(path, thinker)):
                continue
            self._thinkers.append(thinker)

        self._thinkers = sorted(self._thinkers)

        self._epochs = []
        for t in self._thinkers:
            dir = os.path.join(path, t)
            for f in os.listdir(dir):
                p = os.path.join(dir, f)
                if os.path.splitext(p)[-1] != '.npz':
                    continue

                self._epochs.append(p)

    def __len__(self):
        return len(self._epochs)

    def __getitem__(self, idx):
        d = np.load(self._epochs[idx])
        return torch.tensor(d['X']), torch.tensor(d['y'])

    @property
    def person_id(self):
        assert len(self._thinkers) == 1
        return self._thinkers[0]

    def lmso(self, folds):
        group_size = (len(self._thinkers) + folds - 1) // folds
        groups = []
        for i in range(folds):
            low = i * group_size
            high = min(len(self._thinkers), (i + 1) * group_size)
            groups.append(self._thinkers[low:high])

        for val_idx in range(len(groups)):
            test_idx = (val_idx + 1) % len(groups)

            train_thinkers = []
            for i in range(len(groups)):
                if i == val_idx or i == test_idx:
                    continue
                train_thinkers.extend(groups[i])

            train = PreprocessedDataset(self._path, train_thinkers)
            val = PreprocessedDataset(self._path, groups[val_idx])
            test = PreprocessedDataset(self._path, groups[test_idx])

            yield train, val, test

    def loso(self):
        for val_idx in range(len(self._thinkers)):
            test_idx = (val_idx + 1) % len(self._thinkers)

            train_thinkers = []
            for i in range(len(self._thinkers)):
                if i == val_idx or i == test_idx:
                    continue
                train_thinkers.append(self._thinkers[i])

            train = PreprocessedDataset(self._path, train_thinkers)
            val = PreprocessedDataset(self._path, [self._thinkers[val_idx]])
            test = PreprocessedDataset(self._path, [self._thinkers[test_idx]])

            yield train, val, test
