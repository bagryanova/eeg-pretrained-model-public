from os.path import dirname, abspath, join
import sys

PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(join(PROJECT_ROOT, 'src'))

import gc
import os
import torch
import torch.nn as nn

from eeg.data.preprocessed_dataset import PreprocessedDataset
from eeg.globals import WANDB_PROJECT_NAME, DATA_DIR
from eeg.models.data2vec import Data2Vec2ForPretraining, ConvFeatureExtractionModel, modify_data2vec_config_default
from eeg.models.utils import GenericClassifier, LinearHead, ApplyTo
from eeg.training.impl import Data2VecTrainer
from eeg.training.logger import WanDBLogger
from eeg.utils import seed_everything, loop_dataloader, get_experiment_name, create_balanced_sampler
from eeg.data.eeg_dataset import EEGDataset, SubsequenceDataset


EXPERIMENT_NAME = None

TRAINING_PARAMETERS = {
    'batch_size': 64,
    'training_steps': 400000,
    'select_best': 'min loss',
    'eval_steps': 100000000,
    'learning_rate': 4e-4,
    'repeat_batch': 4,
}


def create_model():
    def modify_config(config):
        modify_data2vec_config_default(config)
        config.repeat_batch = TRAINING_PARAMETERS['repeat_batch']
        config.ema_anneal_end_step = 75000
        config.min_target_var = 0.1
        config.ema_encoder_only = False
        config.instance_norm_target_layer = True

    model = Data2Vec2ForPretraining(modify_config=modify_config)
    model.load_checkpoint('data2vec2-base-audio')
    print('Loaded checkpoint')
    model._model.make_target_model()

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

    return model


def get_logger(job):
    return WanDBLogger(
        WANDB_PROJECT_NAME,
        EXPERIMENT_NAME,
        job,
        config=TRAINING_PARAMETERS,
    )


def fit(dataset):
    params = TRAINING_PARAMETERS
    batch_size = params['batch_size']
    training_steps = params['training_steps']
    logging_steps = 10
    select_best = params['select_best']
    eval_steps = params['eval_steps']
    learning_rate = params['learning_rate']

    model = create_model()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=[0.9, 0.98],
        eps=1e-6,
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
            dataset,
            batch_size=batch_size,
            num_workers=8,
            shuffle=True,
            pin_memory=True,
        ),
        training_steps,
    )

    logger = get_logger('Pretrain')
    trainer = Data2VecTrainer(
        model=model,
        device='cuda:0',
        train_dataloader=train_dataloader,
        val_dataloader=None,
        eval_steps=eval_steps,
        optimizer=optimizer,
        clip_grad_norm=None,
        scheduler=scheduler,
        logger=logger,
        logging_steps=logging_steps,
        select_best=select_best,
        checkpoint_every=25000,
    )

    trainer.train()

    del model
    del trainer
    gc.collect()
    torch.cuda.synchronize()


def create_dataset():
    dir = join(DATA_DIR, 'tuh_eeg_h5')
    files = []
    for f in os.listdir(dir):
        if os.path.splitext(f)[1] == '.h5':
            files.append(join(dir, f))

    datasets = []
    for f in files:
        datasets.append(SubsequenceDataset(
            EEGDataset(f),
            60 * 256,
        ))

    return torch.utils.data.ConcatDataset(datasets)


if __name__ == '__main__':
    EXPERIMENT_NAME = get_experiment_name('003_data2vec_pretrain')

    seed_everything(228)

    dataset = create_dataset()

    import warnings
    warnings.filterwarnings('ignore')

    fit(dataset)
