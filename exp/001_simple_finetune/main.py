from os.path import dirname, abspath, join
import sys

PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(join(PROJECT_ROOT, 'src'))

import argparse
import dn3
import torch
import gc

from multiprocessing import Process

from eeg.data.preprocessed_dataset import PreprocessedDataset
from eeg.data.eeg_dataset import EEGDataset, EpochedClassificationDataset
from eeg.globals import WANDB_PROJECT_NAME, PREPROCESSED_DATA_DIR, DATA_CONFIG_PATH, DATA_DIR, WEIGHTS_DIR
from eeg.models.bendr import BENDRTokenizer, BENDREncoder
from eeg.models.utils import GenericClassifier, LinearHead, ApplyTo
from eeg.models.wav2vec import Wav2Vec2Encoder
from eeg.training.impl import ClassificationTrainer
from eeg.training.logger import WanDBLogger
from eeg.utils import seed_everything, loop_dataloader, \
    get_experiment_name, create_balanced_sampler, read_file


EXPERIMENT_NAME = None

TRAINING_PARAMETERS = {
    'mmidb': {
        'batch_size': 4,
        'training_steps': 7000,
        'select_best': 'max balanced_accuracy',
        'eval_steps': 500,
        'learning_rate': 1e-5,
        'weight_decay': 1e-2,
    },
    'sleep-edf': {
        'batch_size': 64,
        'training_steps': 60000,
        'select_best': 'max accuracy',
        'eval_steps': 2000,
        'learning_rate': 5e-5,
        'weight_decay': 1e-2,
    },
    'sleep-edf-pm-30': {
        'batch_size': 64,
        'training_steps': 30000,
        'select_best': 'max accuracy',
        'eval_steps': 2000,
        'learning_rate': 1e-4,
        'weight_decay': 1e-3,
    }
}


def parse_args():
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('model', type=str)

    args = parser.parse_args()
    '''

    from dataclasses import dataclass

    @dataclass
    class Args:
        dataset: str = 'sleep-edf'
        model: str = 'w2v_b'
        fold: int = None

    args = Args()

    global EXPERIMENT_NAME
    EXPERIMENT_NAME = get_experiment_name(f'001_simple_finetune_{args.dataset}_{args.model}')

    return args


def get_dataset(name):
    if name in ['sleep-edf-pm-30']:
        return EpochedClassificationDataset(
            EEGDataset(join(DATA_DIR, 'sleep-edfx-1.0.0-30min-wake-labels.h5'))
        )

    if name in ['sleep-edf']:
        return PreprocessedDataset(join(PREPROCESSED_DATA_DIR, name))

    data_config = dn3.configuratron.ExperimentConfig(DATA_CONFIG_PATH)
    dataset = data_config.datasets[args.dataset].auto_construct_dataset()
    dataset.add_transform(dn3.transforms.instance.To1020())

    return dataset


def get_logger(job):
    return WanDBLogger(
        WANDB_PROJECT_NAME,
        EXPERIMENT_NAME,
        job,
        config=TRAINING_PARAMETERS,
    )


def create_model(args):
    if args.model == 'w2v_b':
        encoder, dim = Wav2Vec2Encoder('facebook/wav2vec2-base-960h'), 768
    elif args.model == 'w2v_l':
        encoder, dim = Wav2Vec2Encoder('facebook/wav2vec2-large-960h'), 1024
    else:
        raise NotImplementedError()

    num_classes = {
        'sleep-edf': 5,
        'sleep-edf-pm-30': 5,
        'mmidb': 2,
    }[args.dataset]

    return GenericClassifier(
        tokenizer=BENDRTokenizer(in_features=20, encoder_h=512),
        encoder=encoder,
        heads=ApplyTo('cls', 'logits', LinearHead(dim, num_classes)),
    )


def fit_one_fold(args, fold, train, val, test):
    # seed_everything(228)

    params = TRAINING_PARAMETERS[args.dataset]
    batch_size = params['batch_size']
    training_steps = params['training_steps']
    logging_steps = 10
    select_best = params['select_best']
    eval_steps = params['eval_steps']
    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']

    model = create_model(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        learning_rate,
        epochs=1,
        steps_per_epoch=training_steps,
        pct_start=0.1,
    )

    train_dataloader = loop_dataloader(
        torch.utils.data.DataLoader(
            train,
            batch_size=batch_size,
            num_workers=8,
            # sampler=create_balanced_sampler(train),
            shuffle=True,
            pin_memory=True,
        ),
        training_steps,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False,
    )

    logger = get_logger(f'Fold {fold}')
    trainer = ClassificationTrainer(
        model=model,
        device='cuda:0',
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        eval_steps=eval_steps,
        optimizer=optimizer,
        clip_grad_norm=None,
        scheduler=scheduler,
        logger=logger,
        logging_steps=logging_steps,
        select_best=select_best,
    )

    # best_model = trainer.train()
    # torch.save(best_model.state_dict(), f'best_model_{fold}.pt')
    trainer._model.load_state_dict(torch.load(join(WEIGHTS_DIR, f'best_model_{fold}.pt'), map_location='cpu'))
    trainer._best_model = trainer._model
    results = trainer.test_by_thinker(test)
    logger.add_metrics_by_thinker('Metrics by thinker', results)

    del model
    del trainer
    gc.collect()
    torch.cuda.synchronize()

    return results


if __name__ == '__main__':
    args = parse_args()
    print(EXPERIMENT_NAME)
    print()

    seed_everything(228)

    dataset = get_dataset(args.dataset)

    import warnings
    warnings.filterwarnings('ignore')

    results = []
    for fold, (train, val, test) in enumerate(dataset.lmso(folds=5)):
        if args.fold is not None and fold != args.fold:
            continue

        print(f'Processing fold {fold}')
        results.extend(fit_one_fold(args, fold, train, val, test))

    get_logger('Results').add_metrics_by_thinker('Metrics by thinker', results)
