import torch
import numpy as np

from tqdm.auto import tqdm
import sklearn.metrics as skmetrics

from eeg.training.trainer import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    def forward(self, batch):
        return self._model(channels=batch[0].to(self._device))['logits']

    def calculate_loss_and_batch_stats(self, out, batch):
        _, y = batch

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(out, y.to(self._device).long())

        preds = torch.argmax(out, dim=-1).cpu().numpy()

        stats = {
            'loss': loss.item(),
            'batch_size': batch[1].shape[0],
            'predicted_labels': preds.copy(),
            'ground_truth_labels': y.numpy(),
        }

        return loss, stats

    def aggregate_batch_stats(self, batch_stats, test=False):
        preds = np.concatenate([b['predicted_labels'] for b in batch_stats])
        ys = np.concatenate([b['ground_truth_labels'] for b in batch_stats])

        res = {
            'loss': np.mean([b['loss'] for b in batch_stats]),
            'accuracy': np.mean(preds == ys),
            'balanced_accuracy': skmetrics.balanced_accuracy_score(ys, preds),
            'cohen_cappa': skmetrics.cohen_kappa_score(ys, preds),
        }

        if test:
            res['predicted_labels'] = preds
            res['ground_truth_labels'] = ys

        return res


class Data2VecTrainer(BaseTrainer):
    def forward(self, batch):
        res = self._model(channels=batch['eeg'].to(self._device))
        self._model.set_num_updates(self._step)
        return res

    def calculate_loss_and_batch_stats(self, out, batch):
        loss = torch.sum(out['losses']['AUDIO_regression'])
        stats = {
            'loss': loss.item(),
            'pred_var': out['pred_var'].item(),
            'target_var': out['target_var'].item(),
        }

        return loss, stats

    def aggregate_batch_stats(self, batch_stats):
        return {
            'loss': np.mean([b['loss'] for b in batch_stats]),
            'pred_var': np.mean([b['pred_var'] for b in batch_stats]),
            'target_var': np.mean([b['target_var'] for b in batch_stats]),
        }
