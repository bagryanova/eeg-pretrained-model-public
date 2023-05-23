import argparse
import numpy as np
import os
import pandas as pd
import sklearn.metrics as M
import warnings

from collections import defaultdict
from functools import partial
from pathlib import Path
from tqdm.auto import tqdm


BOOTSTRAP_ITER = 1000
CONFIDENCE = 0.95

METRICS = {
    'accuracy': M.accuracy_score,
    'balanced_accuracy': M.balanced_accuracy_score,
    'cohen_kappa': M.cohen_kappa_score,
    'macro_f1': partial(M.f1_score, average='macro')
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)

    return parser.parse_args()


def discover_files(dir, pattern):
    d = Path(dir)
    return list(map(str, d.rglob(pattern)))


if __name__ == '__main__':
    args = parse_args()

    if os.path.isdir(args.input):
        df = pd.concat([pd.read_csv(f) for f in discover_files(args.input, '*.csv')])
    else:
        df = pd.read_csv(args.input)
    df = df.reset_index()

    res = defaultdict(list)

    warnings.filterwarnings('ignore')

    print(f'Using {BOOTSTRAP_ITER} iterations and {CONFIDENCE} confidence level')

    for i in tqdm(range(BOOTSTRAP_ITER)):
        idx = np.random.choice(df.shape[0], df.shape[0], replace=True)

        cur = df.loc[idx].copy()

        for name, metric in METRICS.items():
            res[f'{name}_overall'].append(metric(cur['predicted_labels'], cur['ground_truth_labels']))

        cur = cur.groupby('thinker', as_index=False).agg({
            'predicted_labels': list,
            'ground_truth_labels': list,
        })

        for name, metric in METRICS.items():
            by_thinker = []
            for pred, gt in zip(cur['predicted_labels'], cur['ground_truth_labels']):
                by_thinker.append(metric(gt, pred))

            res[f'{name}_mean_by_thinker'].append(np.mean(by_thinker))

    names = []
    means = []
    lows = []
    highs = []
    stds = []
    for k, v in res.items():
        print(f'{k}:')
        mean = np.mean(v)
        low = np.quantile(v, (1 - CONFIDENCE) / 2)
        high = np.quantile(v, 1 - (1 - CONFIDENCE) / 2)
        std = np.std(v)

        names.append(k)
        means.append(mean)
        lows.append(low)
        highs.append(high)
        stds.append(std)

        print(f'Mean: {mean}\nLow:  {low}\nHigh: {high}\nSTD:  {std}')
        print()

    res = pd.DataFrame()
    res['metric'] = names
    res['mean'] = means
    res['low'] = lows
    res['high'] = highs
    res['std'] = stds

    res.to_csv(args.output)
