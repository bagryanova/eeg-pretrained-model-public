from os.path import dirname, abspath, join
import sys

PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(join(PROJECT_ROOT, 'src'))

from eeg.data.eeg_dataset import EEGDataset, EpochedClassificationDataset
from eeg.globals import DATA_DIR

if __name__ == '__main__':
    dataset = EpochedClassificationDataset(
        EEGDataset(join(DATA_DIR, 'sleep-edfx-1.0.0-30min-wake-labels.h5'))
    )

    for train, val, test in dataset.lmso(5):
        print(len(train), len(val), len(test))
