import h5py
import numpy as np
import torch


class EEGDataset:
    _INIT_SIZE = 1000
    _EEG_IND_NAME = '_ind'

    def __init__(self, path, open_type='r'):
        self._path = path
        self._open_type = open_type
        self._f = h5py.File(self._path, self._open_type)

    def initialize(self, items):
        for name, params in items.items():
            dataset = self._f.create_dataset(
                name=name,
                shape=(self._INIT_SIZE,) + params['sample_shape'],
                chunks=params['chunks'],
                maxshape=(None,) + params['sample_shape'],
                dtype=params['type'],
                compression=params['compression'],
            )
            dataset.attrs['len'] = 0

        dataset = self._f.create_dataset(
            name=self._EEG_IND_NAME,
            shape=(self._INIT_SIZE, 2),
            maxshape=(None, 2),
            dtype=np.int64,
        )
        dataset.attrs['len'] = 0

    def append(self, data):
        def append(dataset, sample):
            sample_len = v.shape[0] if len(v.shape) == len(dataset.shape) else 1
            self._resize_maybe(dataset, sample_len)
            dataset[dataset.attrs['len']:(dataset.attrs['len'] + sample_len)] = sample
            dataset.attrs['len'] += sample_len

        eeg_len_before = self._f['eeg'].attrs['len']
        for k, v in data.items():
            if not isinstance(v, np.ndarray):
                v = np.array(v)
            append(self._f[k], v)

        eeg_len_after = self._f['eeg'].attrs['len']
        append(self._f[self._EEG_IND_NAME], np.array([eeg_len_before, eeg_len_after]))

    @property
    def file(self):
        return self._f

    def reset(self):
        self._f = None
        return self

    def open(self):
        if self._f is not None:
            return
        self._f = h5py.File(self._path, self._open_type)

    def _resize_maybe(self, dataset, add_size):
        dataset_length = dataset.attrs['len']
        while dataset_length + add_size > dataset.shape[0]:
            dataset.resize((dataset.shape[0] * 2,) + dataset.shape[1:])


class SubsequenceDataset(torch.utils.data.Dataset):
    def __init__(self, eeg_dataset, subsequence_length):
        self._dataset = eeg_dataset
        self._dataset.open()

        self._ranges = []
        ind_data = self._dataset.file[EEGDataset._EEG_IND_NAME]
        for i, r in enumerate(ind_data[:ind_data.attrs['len']]):
            cnt = (r[1] - r[0]) // subsequence_length
            self._ranges.append(
                np.vstack([
                    np.arange(cnt) * subsequence_length + r[0],
                    (np.arange(cnt) + 1) * subsequence_length + r[0],
                    np.ones(cnt, dtype=np.int64) * i,
                ]).T.copy()
            )
        self._ranges = np.concatenate(self._ranges, axis=0)

    def __getitem__(self, idx):
        self._dataset.open()
        first, last, ind = self._ranges[idx]
        return {
            'eeg': self._dataset.file['eeg'][first:last].T,
        }

    def __len__(self):
        return self._ranges.shape[0]

    def reset(self):
        self._dataset.reset()
        return self


class EpochedDataset(torch.utils.data.Dataset):
    def __len__(self):
        return self._sub_indices.shape[0]

    def reset(self):
        self._dataset.reset()
        return self

    def lmso(self, folds, shuffle=False):
        thinkers = self.thinkers.copy()

        if shuffle:
            np.random.shuffle(thinkers)

        group_size = (len(thinkers) + folds - 1) // folds
        groups = []
        for i in range(folds):
            low = i * group_size
            high = min(len(thinkers), (i + 1) * group_size)
            groups.append(thinkers[low:high])

        for val_idx in range(len(groups)):
            test_idx = (val_idx + 1) % len(groups)

            train_thinkers = []
            for i in range(len(groups)):
                if i == val_idx or i == test_idx:
                    continue
                train_thinkers.extend(groups[i])

            train = self._subset(self._get_indices_for_thinkers(train_thinkers))
            val = self._subset(self._get_indices_for_thinkers(groups[val_idx]))
            test = self._subset(self._get_indices_for_thinkers(groups[test_idx]))

            yield train, val, test

    def loso(self):
        yield from self.lmso(folds=len(np.unique(self.thinkers)))

    @property
    def thinkers(self):
        return np.unique(self._thinkers[self._sub_indices])

    @property
    def person_id(self):
        thinkers = self.thinkers
        assert len(thinkers) == 1
        return int(thinkers[0])

    def _get_indices_for_thinkers(self, thinkers):
        s = set(thinkers)

        res = []
        for i in self._sub_indices:
            if self._thinkers[i] in s:
                res.append(i)

        res = np.array(res)
        return res


class EpochedClassificationDataset(EpochedDataset):
    def __init__(self, eeg_dataset, sub_indices=None):
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

    def __getitem__(self, idx):
        self._dataset.open()
        X = self._dataset.file['eeg'][self._sub_indices[idx]].T
        y = self._dataset.file['label'][self._sub_indices[idx]]
        return X, y

    def _subset(self, ind):
        return EpochedClassificationDataset(self._dataset, ind)
