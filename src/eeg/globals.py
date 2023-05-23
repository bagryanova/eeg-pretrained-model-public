from os.path import dirname, abspath, join

PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__))))

WANDB_PROJECT_NAME = 'eeg'

DATA_DIR = join(PROJECT_ROOT, '.data')
PREPROCESSED_DATA_DIR = join(DATA_DIR, 'preprocessed')

WEIGHTS_DIR = join(PROJECT_ROOT, 'weights')

DATA_CONFIG_PATH = join(PROJECT_ROOT, 'exp', 'dn3_configs', 'datasets.yml')

EEG_1020_CHANNELS = [
            'FP1', 'FP2',
    'F7', 'F3', 'FZ', 'F4', 'F8',
    'T7', 'C3', 'CZ', 'C4', 'T8',
    'T5', 'P3', 'PZ', 'P4', 'T6',
            'O1', 'O2'
]
