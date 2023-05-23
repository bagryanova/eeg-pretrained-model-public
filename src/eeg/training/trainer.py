import copy
import torch
import numpy as np

from tqdm.auto import tqdm


class BaseTrainer:
    def __init__(
        self,
        model,
        device,
        train_dataloader,
        val_dataloader,
        eval_steps,
        optimizer,
        clip_grad_norm,
        scheduler,
        logger,
        logging_steps,
        select_best,
        checkpoint_every=None):

        self._model = model
        self._device = device

        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._eval_steps = eval_steps

        self._optimizer = optimizer
        self._clip_grad_norm = clip_grad_norm
        self._scheduler = scheduler

        self._logger = logger
        self._logging_steps = logging_steps

        self._step = 0

        self._best_model = None
        self._best_metric = None
        compare, self._select_best_metric = select_best.split()
        assert compare in ['min', 'max']
        self._maximize = compare == 'max'

        self._checkpoint_every = checkpoint_every

    def train(self):
        self._model.to(self._device)

        self._batch_stats = []
        for batch in tqdm(self._train_dataloader):
            self._train_step(batch)
            if self._step > 0 and self._step % self._eval_steps == 0:
                self._eval()

            if self._checkpoint_every is not None:
                if self._step > 0 and self._step % self._checkpoint_every == 0:
                    self._checkpoint()

        if self._val_dataloader is not None:
            self._eval()

        if self._checkpoint_every is not None:
            self._checkpoint()

        return self._best_model

    def test_by_thinker(self, dataset):
        self._model.cpu()
        self._model.load_state_dict(self._best_model.state_dict())

        results = []
        for _, _, thinker in dataset.loso():
            dataloader = torch.utils.data.DataLoader(thinker, batch_size=16, shuffle=False)
            cur = self._inference(dataloader, test=True)
            cur['thinker'] = str(thinker.person_id)
            results.append(cur)

        return results

    def forward(self, batch):
        raise NotImplementedError()

    def calculate_loss_and_batch_stats(self, out, batch):
        raise NotImplementedError()

    def aggregate_batch_stats(batch_stats):
        raise NotImplementedError()

    def _train_step(self, batch):
        self._model.train()

        self._logger.set_step(self._step)
        self._optimizer.zero_grad()

        out = self.forward(batch)
        loss, cur_batch_stats = self.calculate_loss_and_batch_stats(out, batch)

        if torch.isnan(loss):
            print('ERROR: loss is NaN')
            raise ValueError('Loss is NaN')

        loss.backward()
        if self._clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_grad_norm)

        self._optimizer.step()
        self._step += 1

        self._batch_stats.append(cur_batch_stats)

        if len(self._batch_stats) >= self._logging_steps:
            self._log(self.aggregate_batch_stats(self._batch_stats))
            self._batch_stats = []
            self._log_learning_rate()

        self._scheduler.step()

    def _eval(self):
        self._logger.set_step(self._step, mode='eval')
        results = self._inference(self._val_dataloader)
        self._log(results)

        print('Validation results:')
        for k, v in results.items():
            print(f'{k}: {v}')
        print()

        # import time
        # for _ in tqdm(range(300)):
        #     time.sleep(1.0)

        self._update_best_maybe(results)

    @torch.no_grad()
    def _inference(self, dataloader, test=False):
        self._model.to(self._device).eval()

        batch_stats = []
        for batch in tqdm(dataloader):
            out = self.forward(batch)
            _, cur_batch_stats = self.calculate_loss_and_batch_stats(out, batch)

            batch_stats.append(cur_batch_stats)

        return self.aggregate_batch_stats(batch_stats, test=test)

    def _log(self, results):
        for k, v in results.items():
            self._logger.add_scalar(k, v)

        return results

    def _log_learning_rate(self):
        for param_group in self._optimizer.param_groups:
            self._logger.add_scalar('learning_rate', param_group['lr'])
            break

    def _update_best_maybe(self, val_results):
        def update():
            self._best_model = copy.deepcopy(self._model.cpu())
            self._model.to(self._device)
            self._best_metric = val_results[self._select_best_metric]
            print(f'New best val metric: {self._best_metric}')

        if self._best_metric is None:
            update()
            return

        is_better = self._best_metric > val_results[self._select_best_metric]
        if self._maximize:
            is_better = not is_better

        if not is_better:
            return

        update()

    def _checkpoint(self):
        path = f'checkpoint_{self._step}.pt'
        torch.save(
            {
                'step': self._step,
                'model_state_dict': self._model.state_dict(),
                'optimizer_state_dict': self._optimizer.state_dict(),
            },
            path,
        )
