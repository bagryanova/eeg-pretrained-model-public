from os.path import dirname, abspath, join
import sys

PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(join(PROJECT_ROOT, 'src'))

import dn3
import torch
import gc

from multiprocessing import Process

from eeg.globals import WANDB_PROJECT_NAME
from eeg.models.bendr import BENDRTokenizer, BENDREncoder
from eeg.models.utils import GenericClassifier, LinearHead, ApplyTo
from eeg.models.wav2vec import Wav2Vec2Encoder
from eeg.training.impl import ClassificationTrainer
from eeg.training.logger import WanDBLogger
from eeg.utils import seed_everything, loop_dataloader, get_experiment_name


EXPERIMENT_NAME = get_experiment_name('000_test')


def get_logger(job):
    return WanDBLogger(WANDB_PROJECT_NAME, EXPERIMENT_NAME, job)


def fit_one_fold(fold, train, val, test):
    # seed_everything(228)

    batch_size = 4
    training_steps = 7000
    logging_steps = 10
    select_best = 'max balanced_accuracy'
    eval_steps = 600
    learning_rate = 1e-5

    model = GenericClassifier(
        tokenizer=BENDRTokenizer(in_features=20, encoder_h=512).load_checkpoint('bendr-tokenizer-pretrained'),
        encoder=Wav2Vec2Encoder('facebook/wav2vec2-base-960h'),
        heads=ApplyTo('cls', 'logits', LinearHead(768, 2)),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
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
            num_workers=4,
            shuffle=True,
        ),
        training_steps,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val,
        batch_size=batch_size,
        num_workers=4,
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
        logger=get_logger(f'Fold {fold}'),
        logging_steps=logging_steps,
        select_best=select_best,
    )

    best_model = trainer.train()
    results = trainer.test_by_thinker(test)

    del model
    del trainer
    gc.collect()
    torch.cuda.synchronize()

    return results


if __name__ == '__main__':
    seed_everything(228)

    data_config = dn3.configuratron.ExperimentConfig(join(PROJECT_ROOT, 'exp', 'dn3_configs', 'datasets.yml'))

    dataset = data_config.datasets['mmidb'].auto_construct_dataset()
    dataset.add_transform(dn3.transforms.instance.To1020())

    import warnings
    warnings.filterwarnings('ignore')

    results = []
    for fold, (train, val, test) in enumerate(dataset.lmso(folds=5)):
        results.extend(fit_one_fold(fold, train, val, test))

    get_logger('Results').add_metrics_by_thinker('Metrics by thinker', results)
