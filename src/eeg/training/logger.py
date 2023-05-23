import numbers
import numpy as np
import pandas as pd
import wandb
import copy

from collections import defaultdict
from datetime import datetime


class WanDBLogger:
    def __init__(self, project_name, experiment_name, job_name, config=None, run_id=None):
        self.writer = None
        self.selected_module = ''

        try:
            import wandb
            wandb.login()

            wandb.init(
                project=project_name,
                group=experiment_name,
                job_type=job_name,
                reinit=True,
                config=config,
                id=run_id,
            )
            wandb.Table.MAX_ROWS = 1000000
            self.wandb = wandb

        except ImportError:
            print('To use wandb install it via \n\t pip install wandb')

        self.step = 0
        self.mode = ''
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def _scalar_name(self, scalar_name):
        return f'{scalar_name}_{self.mode}'

    def add_scalar(self, scalar_name, scalar):
        self.wandb.log({
            self._scalar_name(scalar_name): scalar,
        }, step=self.step)

    def add_scalars(self, tag, scalars):
        self.wandb.log({
            **{f'{scalar_name}_{tag}_{self.mode}': scalar for scalar_name, scalar in
               scalars.items()}
        }, step=self.step)

    def add_image(self, scalar_name, image):
        self.wandb.log({
            self._scalar_name(scalar_name): self.wandb.Image(image)
        }, step=self.step)

    def add_audio(self, scalar_name, audio, sample_rate=None):
        audio = audio.detach().cpu().numpy().T
        self.wandb.log({
            self._scalar_name(scalar_name): self.wandb.Audio(audio, sample_rate=sample_rate)
        }, step=self.step)

    def add_text(self, scalar_name, text):
        self.wandb.log({
            self._scalar_name(scalar_name): self.wandb.Html(text)
        }, step=self.step)

    def add_histogram(self, scalar_name, hist, bins=None):
        hist = hist.detach().cpu().numpy()
        np_hist = np.histogram(hist, bins=bins)
        if np_hist[0].shape[0] > 512:
            np_hist = np.histogram(hist, bins=512)

        hist = self.wandb.Histogram(
            np_histogram=np_hist
        )

        self.wandb.log({
            self._scalar_name(scalar_name): hist
        }, step=self.step)

    def add_table(self, table_name, table: pd.DataFrame):
        self.wandb.log({self._scalar_name(table_name): wandb.Table(dataframe=table)},
                       step=self.step)

    def add_metrics_by_thinker(self, table_name, results):
        table = pd.DataFrame()

        arrays_thinkers = []
        arrays = defaultdict(list)

        mean = dict()
        for name in ['thinker'] + list(results[0].keys()):
            col = list([t[name] for t in results])

            if name == 'thinker':
                arrays_thinkers = np.array(col)
                col = ['mean'] + col
            elif isinstance(col[0], numbers.Number):
                mean[name] = np.mean(col)
                col = [np.mean(col)] + col
            elif isinstance(col[0], np.ndarray):
                arrays[name] = copy.deepcopy(col)
                continue
            else:
                col = [None] + col

            table[name] = col

        self.add_table(table_name, table)

        if len(arrays) > 0:
            arrays_df = pd.DataFrame()
            sz = list()
            for k, v in arrays.items():
                if len(sz) == 0:
                    for i, a in enumerate(v):
                        sz.append(a.shape[0])
                arrays_df[k] = np.concatenate(v)

            thinkers = []
            for s, thinker in zip(sz, arrays_thinkers):
                thinkers.extend([thinker] * s)

            arrays_df['thinker'] = thinkers
            self.add_table(f'{table_name}_arrays', arrays_df)

        self.mode = 'test'
        for k, v in mean.items():
            self.add_scalar(k, v)

    def add_images(self, scalar_name, images):
        raise NotImplementedError()

    def add_pr_curve(self, scalar_name, scalar):
        raise NotImplementedError()

    def add_embedding(self, scalar_name, scalar):
        raise NotImplementedError()
