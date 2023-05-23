from os.path import dirname, abspath, join
import sys

PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(join(PROJECT_ROOT, 'src'))

import argparse
import copy
import gc
import torch
import torch.nn as nn
import wandb

from names_generator import generate_name
from pprint import pprint
from functools import partial

from eeg.data.preprocessed_dataset import PreprocessedDataset
from eeg.globals import WANDB_PROJECT_NAME, DATA_DIR, WEIGHTS_DIR
from eeg.models.data2vec import Data2Vec2ForClassification, ConvFeatureExtractionModel, modify_data2vec_config_default
from eeg.models.utils import GenericClassifier, LinearHead, ApplyTo, AddCLSToken, IgnoreFirstToken
from eeg.training.impl import ClassificationTrainer
from eeg.training.logger import WanDBLogger
from eeg.utils import seed_everything, loop_dataloader, \
    get_experiment_name, create_balanced_sampler, generate_name_seed_agnostic
from eeg.data.eeg_dataset import EEGDataset, EpochedClassificationDataset, SubsequenceDataset

from eeg.globals import WANDB_PROJECT_NAME


TRAINING_PARAMETERS = {
    'batch_size': 64,
    'training_steps': 30000,
    'select_best': 'max accuracy',
    'eval_steps': 2000,
    'learning_rate': 1e-4,
    'weight_decay': 1e-3,
}


def get_logger(job, group):
    return WanDBLogger(
        WANDB_PROJECT_NAME,
        group,
        job,
    )


def create_model(instance_norm):
    model = Data2Vec2ForClassification(modify_config=modify_data2vec_config_default)

    model.change_conv_network(
        nn.Sequential(
            nn.InstanceNorm1d(19, eps=1e-9) if instance_norm else nn.Identity(),
            nn.Identity(),
            ConvFeatureExtractionModel(
                [(512, 10, 5)] + [(512, 3, 2)] * 4,
                in_channels=19,
            ),
        )
    )

    model.load_checkpoint(join(WEIGHTS_DIR, 'data2vec_base_eeg_400000.pt'))

    model = GenericClassifier(
        tokenizer=model,
        encoder=None,
        heads=ApplyTo('cls', 'logits', LinearHead(768, 5)),
    )

    model._tokenizer._model.modality_encoders['AUDIO'].relative_positional_encoder = IgnoreFirstToken(
        model._tokenizer._model.modality_encoders['AUDIO'].relative_positional_encoder,
    )

    model._tokenizer._model.modality_encoders['AUDIO'].project_features = AddCLSToken(
        model._tokenizer._model.modality_encoders['AUDIO'].project_features,
        768,
    )

    return model


def get_data():
    dataset = EpochedClassificationDataset(
        EEGDataset(join(DATA_DIR, 'sleep-edfx-1.0.0-30min-wake-labels.h5')),
    )

    for fold, (train, val, test) in enumerate(dataset.lmso(folds=5, shuffle=True)):
        return train, val, test


def do_train(sweep_id):
    group_name = f'sweep_{sweep_id}_{generate_name_seed_agnostic()}'

    seed_everything(228)

    train, val, test = get_data()

    logger = get_logger('sweep_run', group_name)
    params = copy.deepcopy(TRAINING_PARAMETERS)
    params.update(wandb.config)

    print('Training parameters:')
    pprint(params)
    print()
    print()

    batch_size = params['batch_size']
    training_steps = params['training_steps']
    logging_steps = 10
    select_best = params['select_best']
    eval_steps = params['eval_steps']
    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']

    model = create_model(params['instance_norm'])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
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
            # sampler=create_balanced_sampler(train, num_workers=0),
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

    trainer.train()
    results = trainer.test_by_thinker(test)
    logger.add_metrics_by_thinker('Metrics by thinker', results)

    del model
    del trainer
    del optimizer
    gc.collect()
    torch.cuda.synchronize()


if __name__ == '__main__':
    with open('.sweep_id', 'r') as f:
        sweep_id = f.read().strip()

    wandb.agent(
        sweep_id,
        function=partial(do_train, sweep_id),
        project=WANDB_PROJECT_NAME,
    )
