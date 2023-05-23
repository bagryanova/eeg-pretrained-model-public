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
import dn3

from pprint import pprint
from processify import processify
from functools import partial

from eeg.data.preprocessed_dataset import PreprocessedDataset
from eeg.globals import WANDB_PROJECT_NAME, WEIGHTS_DIR, DATA_CONFIG_PATH
from eeg.models.data2vec import Data2Vec2ForClassification, ConvFeatureExtractionModel, modify_data2vec_config_default
from eeg.models.utils import GenericClassifier, LinearHead, ApplyTo, AddCLSToken, IgnoreFirstToken
from eeg.training.impl import ClassificationTrainer
from eeg.training.logger import WanDBLogger
from eeg.utils import seed_everything, loop_dataloader, generate_name_seed_agnostic, reset_wandb_env
from eeg.data.eeg_dataset import EEGDataset, EpochedClassificationDataset, SubsequenceDataset

from eeg.globals import WANDB_PROJECT_NAME


TRAINING_PARAMETERS = {
    'batch_size': 8,
    'training_steps': 5000,
    'select_best': 'max accuracy',
    'eval_steps': 250,
    'learning_rate': 1e-4,
    'weight_decay': 0,
}


def get_logger(job, group):
    return WanDBLogger(
        WANDB_PROJECT_NAME,
        group,
        job,
    )


def create_model():
    model = Data2Vec2ForClassification(modify_config=modify_data2vec_config_default)

    model.change_conv_network(
        nn.Sequential(
            nn.InstanceNorm1d(19, eps=1e-9),
            nn.Dropout1d(p=0.3),
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
        heads=ApplyTo('cls', 'logits', LinearHead(768, 2)),
    )

    model._tokenizer._model.modality_encoders['AUDIO'].relative_positional_encoder = IgnoreFirstToken(
        model._tokenizer._model.modality_encoders['AUDIO'].relative_positional_encoder,
    )

    model._tokenizer._model.modality_encoders['AUDIO'].project_features = AddCLSToken(
        model._tokenizer._model.modality_encoders['AUDIO'].project_features,
        768,
    )

    return model


@processify
def train_one_fold(group, config, fold, train, val, test):
    reset_wandb_env()

    logger = get_logger(f'fold_{fold}', group)
    params = copy.deepcopy(TRAINING_PARAMETERS)
    params.update(config)

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

    model = create_model()

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

    del model
    del trainer
    del optimizer
    gc.collect()
    torch.cuda.synchronize()

    return results


def do_train(sweep_id):
    group_name = f'sweep_{sweep_id}_{generate_name_seed_agnostic()}'
    print('!!!', group_name)
    print()
    print()

    sweep_logger = get_logger('results', group_name)
    config = wandb.config

    seed_everything(228)

    data_config = dn3.configuratron.ExperimentConfig(DATA_CONFIG_PATH)
    dataset = data_config.datasets['mmidb'].auto_construct_dataset()
    dataset.add_transform(dn3.transforms.instance.To1020(include_scale_ch=False))

    results = []
    for fold, (train, val, test) in enumerate(dataset.lmso(folds=5)):
        print(f'Processing fold {fold}')
        results.extend(train_one_fold(group_name, config, fold, train, val, test))

    sweep_logger.add_metrics_by_thinker('Metrics by thinker', results)


if __name__ == '__main__':
    with open('.sweep_id', 'r') as f:
        sweep_id = f.read().strip()

    wandb.agent(sweep_id, function=partial(do_train, sweep_id=sweep_id), project=WANDB_PROJECT_NAME)
