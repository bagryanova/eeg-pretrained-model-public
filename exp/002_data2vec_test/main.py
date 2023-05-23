from os.path import dirname, abspath, join
import sys

PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(join(PROJECT_ROOT, 'src'))

import argparse
import gc
import torch
import torch.nn as nn

from eeg.data.preprocessed_dataset import PreprocessedDataset
from eeg.globals import WANDB_PROJECT_NAME, DATA_DIR, PREPROCESSED_DATA_DIR
from eeg.models.data2vec import Data2Vec2ForClassification, ConvFeatureExtractionModel, modify_data2vec_config_default
from eeg.models.utils import GenericClassifier, LinearHead, ApplyTo, AddCLSToken, IgnoreFirstToken
from eeg.training.impl import ClassificationTrainer
from eeg.training.logger import WanDBLogger
from eeg.utils import seed_everything, loop_dataloader, get_experiment_name, create_balanced_sampler
from eeg.data.eeg_dataset import EEGDataset, EpochedClassificationDataset, SubsequenceDataset


EXPERIMENT_NAME = None

TRAINING_PARAMETERS = {
    'batch_size': 4,
    'training_steps': 10000,
    'select_best': 'max accuracy',
    'eval_steps': 500,
    'learning_rate': 2e-5,
    'weight_decay': 1e-2,
}


def create_model(checkpoint):
    model = Data2Vec2ForClassification(modify_config=modify_data2vec_config_default)

    if checkpoint == 'data2vec2-base-audio':
        model.load_checkpoint(checkpoint)

    class SelectChannels(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[:, :19, :]

    model.change_conv_network(
        nn.Sequential(
            nn.InstanceNorm1d(19, eps=1e-9),
            nn.Dropout1d(p=0.5), # SelectChannels(),
            ConvFeatureExtractionModel(
                [(512, 10, 5)] + [(512, 3, 2)] * 4,
                in_channels=19,
            ),
        )
    )

    if checkpoint != 'data2vec2-base-audio' and checkpoint != 'none':
        model.load_checkpoint(checkpoint)

    model = GenericClassifier(
        tokenizer=model,
        encoder=None,
        heads=ApplyTo('cls', 'logits', LinearHead(768, 2)),
    )

    # for p in model._tokenizer.parameters():
    #     p.requires_grad = False

    model._tokenizer._model.modality_encoders['AUDIO'].relative_positional_encoder = IgnoreFirstToken(
        model._tokenizer._model.modality_encoders['AUDIO'].relative_positional_encoder,
    )

    model._tokenizer._model.modality_encoders['AUDIO'].project_features = AddCLSToken(
        model._tokenizer._model.modality_encoders['AUDIO'].project_features,
        768,
    )

    return model


def get_logger(job):
    return WanDBLogger(
        WANDB_PROJECT_NAME,
        EXPERIMENT_NAME,
        job,
        config=TRAINING_PARAMETERS,
    )


def fit_one_fold(args, fold, train, val, test):
    params = TRAINING_PARAMETERS
    batch_size = params['batch_size']
    training_steps = params['training_steps']
    logging_steps = 10
    select_best = params['select_best']
    eval_steps = params['eval_steps']
    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']

    model = create_model(args.checkpoint)

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
            sampler=create_balanced_sampler(train, num_workers=12),
            # shuffle=True,
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

    best_model = trainer.train()
    torch.save(best_model.state_dict(), f'best_model_{fold}.pt')
    results = trainer.test_by_thinker(test)
    logger.add_metrics_by_thinker('Metrics by thinker', results)

    del model
    del trainer
    gc.collect()
    torch.cuda.synchronize()

    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    EXPERIMENT_NAME = get_experiment_name(f'002_data2vec_test_sleepedf_old')

    seed_everything(228)

    dataset = EpochedClassificationDataset(
        EEGDataset(join(DATA_DIR, 'eegmat.h5')),
    )
    # dataset = EpochedClassificationDataset(
    #     EEGDataset(join(DATA_DIR, 'sleep-edfx-1.0.0-30min-wake-labels.h5')),
    # )
    # dataset = PreprocessedDataset(join(PREPROCESSED_DATA_DIR, 'sleep-edf'))

    import warnings
    warnings.filterwarnings('ignore')

    results = []
    for fold, (train, val, test) in enumerate(dataset.lmso(folds=5, shuffle=True)):
        print(f'Processing fold {fold}')
        results.extend(fit_one_fold(args, fold, train, val, test))

    get_logger('Results').add_metrics_by_thinker('Metrics by thinker', results)
