from os.path import dirname, abspath, join
import sys

PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(join(PROJECT_ROOT, 'src'))

import argparse
import os
import yadisk

from tqdm.auto import tqdm

from eeg.globals import WEIGHTS_DIR


FILES = {
    'data2vec_base_audio': 'https://disk.yandex.ru/d/8CMKIK50iz2ldQ',
    'data2vec_base_eeg_250000': 'https://disk.yandex.ru/d/Nxv_W3fx1Ddyeg',
    'data2vec_base_eeg_400000': 'https://disk.yandex.ru/d/HMwW6xT88RyY2A',
    'bendr_tokenizer': 'https://disk.yandex.ru/d/EAC5ceQwwQJAwg',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    link = FILES[args.model]
    d = yadisk.YaDisk()
    d.download_public(link, join(WEIGHTS_DIR, f'{args.model}.pt'))
