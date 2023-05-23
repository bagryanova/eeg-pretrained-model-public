from os.path import dirname, abspath, join
import sys

PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(join(PROJECT_ROOT, 'src'))

import argparse
import wandb

from eeg.globals import WANDB_PROJECT_NAME


SWEEP_CONFIG = {
    'sleep-edf': {
        'method': 'bayes',
        'metric': {
            'goal': 'maximize',
            'name': 'accuracy_test',
        },
        'parameters': {
            'learning_rate': {
                'min': 5e-5,
                'max': 2e-4,
                'distribution': 'log_uniform_values',
            },
            'weight_decay': {
                'min': 1e-7,
                'max': 1e-3,
                'distribution': 'log_uniform_values',
            },
            'instance_norm': {
                'values': [True, False],
            },
        }
    },
    'mmidb': {
        'method': 'bayes',
        'metric': {
            'goal': 'maximize',
            'name': 'accuracy_test',
        },
        'parameters': {
            'learning_rate': {
                'min': 5e-6,
                'max': 5e-4,
                'distribution': 'log_uniform_values',
            },
            'weight_decay': {
                'min': 1e-7,
                'max': 1e-2,
                'distribution': 'log_uniform_values',
            },
            'batch_size': {
                'values': [4, 8, 16],
            },
            # 'training_steps': {
            #     'values': [5000, 10000],
            # },
        }
    }
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    sweep_id = wandb.sweep(
        sweep=SWEEP_CONFIG[args.dataset],
        project=WANDB_PROJECT_NAME
    )

    with open('.sweep_id', 'w') as f:
        f.write(sweep_id + '\n')
