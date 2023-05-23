from os.path import dirname, abspath, join
import sys

PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(join(PROJECT_ROOT, 'src'))

import argparse
import dhedfreader
import h5py
import math
import numpy as np
import os

from mne.io import read_raw_edf
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime

from eeg.data.eeg_dataset import EEGDataset
from eeg.globals import EEG_1020_CHANNELS


TARGET_SR = 100

W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "UNKNOWN": UNKNOWN
}

class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "UNKNOWN"
}

ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}

EPOCH_SEC_SIZE = 30
CHANNEL = 'EEG Fpz-Cz'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)

    return parser.parse_args()


def discover_files(dir, pattern):
    d = Path(dir)
    return list(map(str, d.rglob(pattern)))


def read_pair(eeg_file, annotation_file):
    raw = read_raw_edf(eeg_file, preload=True, stim_channel=None)
    # raw = raw.resample(TARGET_SR, n_jobs=6)
    raw_ch_df = raw.to_data_frame()[CHANNEL]
    raw_ch_df = raw_ch_df.to_frame()
    raw_ch_df.set_index(np.arange(len(raw_ch_df)))

    # Get raw header
    with open(eeg_file, 'r', encoding='iso-8859-1') as f:
        reader_raw = dhedfreader.BaseEDFReader(f)
        reader_raw.read_header()
        h_raw = reader_raw.header
    raw_start_dt = datetime.strptime(h_raw['date_time'], "%Y-%m-%d %H:%M:%S")

    # Read annotation and its header
    with open(annotation_file, 'r', encoding='iso-8859-1') as f:
        reader_ann = dhedfreader.BaseEDFReader(f)
        reader_ann.read_header()
        h_ann = reader_ann.header
        _, _, ann = list(zip(*reader_ann.records()))
    ann_start_dt = datetime.strptime(h_ann['date_time'], "%Y-%m-%d %H:%M:%S")

    # Assert that raw and annotation files start at the same time
    assert raw_start_dt == ann_start_dt

    return raw_ch_df, ann[0]


def make_epochs(eeg, ann):
    remove_idx = []    # indicies of the data that will be removed
    labels = []        # indicies of the data that have labels
    label_idx = []
    for a in ann:
        onset_sec, duration_sec, ann_char = a
        ann_str = "".join(ann_char)
        label = ann2label[ann_str]
        if label != UNKNOWN:
            if duration_sec % EPOCH_SEC_SIZE != 0:
                raise Exception("Something wrong")
            duration_epoch = int(duration_sec / EPOCH_SEC_SIZE)
            label_epoch = np.ones(duration_epoch, dtype=int) * label
            labels.append(label_epoch)
            idx = int(onset_sec * TARGET_SR) + np.arange(duration_sec * TARGET_SR, dtype=int)
            label_idx.append(idx)
        else:
            idx = int(onset_sec * TARGET_SR) + np.arange(duration_sec * TARGET_SR, dtype=int)
            remove_idx.append(idx)

    labels = np.hstack(labels)

    print("before remove unwanted: {}".format(np.arange(len(eeg)).shape))
    if len(remove_idx) > 0:
        remove_idx = np.hstack(remove_idx)
        select_idx = np.setdiff1d(np.arange(len(eeg)), remove_idx)
    else:
        select_idx = np.arange(len(eeg))
    print("after remove unwanted: {}".format(select_idx.shape))

    # Select only the data with labels
    print("before intersect label: {}".format(select_idx.shape))
    label_idx = np.hstack(label_idx)
    select_idx = np.intersect1d(select_idx, label_idx)
    print("after intersect label: {}".format(select_idx.shape))

    if len(label_idx) > len(select_idx):
        print("before remove extra labels: {}, {}".format(select_idx.shape, labels.shape))
        extra_idx = np.setdiff1d(label_idx, select_idx)
        # Trim the tail
        if np.all(extra_idx > select_idx[-1]):
            n_trims = len(select_idx) % int(EPOCH_SEC_SIZE * TARGET_SR)
            n_label_trims = int(math.ceil(n_trims / (EPOCH_SEC_SIZE * TARGET_SR)))
            select_idx = select_idx[:-n_trims]
            labels = labels[:-n_label_trims]
        print("after remove extra labels: {}, {}".format(select_idx.shape, labels.shape))

    # Remove movement and unknown stages if any
    raw_ch = eeg.values[select_idx]

    # Verify that we can split into 30-s epochs
    if len(raw_ch) % (EPOCH_SEC_SIZE * TARGET_SR) != 0:
        raise Exception("Something wrong")
    n_epochs = len(raw_ch) / (EPOCH_SEC_SIZE * TARGET_SR)

    # Get epochs and their corresponding labels
    x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
    y = labels.astype(np.int32)

    assert len(x) == len(y)

    w_edge_mins = 30
    nw_idx = np.where(y != stage_dict["W"])[0]
    start_idx = nw_idx[0] - (w_edge_mins * 2)
    end_idx = nw_idx[-1] + (w_edge_mins * 2)
    if start_idx < 0: start_idx = 0
    if end_idx >= len(y): end_idx = len(y) - 1
    select_idx = np.arange(start_idx, end_idx+1)
    print(("Data before selection: {}, {}".format(x.shape, y.shape)))
    x = x[select_idx]
    y = y[select_idx]
    print(("Data after selection: {}, {}".format(x.shape, y.shape)))

    X = np.zeros(x.shape[:2] + (len(EEG_1020_CHANNELS),), dtype=np.float32)
    X[:, :, EEG_1020_CHANNELS.index('PZ')] = x.squeeze()

    return X, y


if __name__ == '__main__':
    args = parse_args()

    dataset = EEGDataset(args.output, '2')
    del dataset.file['label']
    dataset.initialize({
        'eeg': {
            'sample_shape': (30 * TARGET_SR, 19,),
            'chunks': (1, 30 * TARGET_SR, 19),
            'type': np.float32,
            'compression': 'gzip',
        },
        'filename': {
            'sample_shape': tuple(),
            'chunks': True,
            'type': h5py.special_dtype(vlen=str),
            'compression': None,
        },
        'thinker': {
            'sample_shape': tuple(),
            'chunks': True,
            'type': np.int32,
            'compression': None,
        },
        'label': {
            'sample_shape': tuple(),
            'chunks': True,
            'type': np.int32,
            'compression': None,
        },
    })

    eeg_files = sorted(discover_files(args.input, '*PSG.edf'))
    annotation_files = sorted(discover_files(args.input, '*Hypnogram.edf'))

    for eeg_file, annotation_file in zip(tqdm(eeg_files), annotation_files):
        try:
            f = os.path.basename(eeg_file)
            thinker = int(f[3:5])

            print(f, thinker)

            eeg, ann = read_pair(eeg_file, annotation_file)
            X, y = make_epochs(eeg, ann)

            dataset.append({
                'eeg': X,
                'filename': f,
                'thinker': thinker,
                'label': y,
            })

        except KeyboardInterrupt:
            break

        except ZeroDivisionError:
            pass
