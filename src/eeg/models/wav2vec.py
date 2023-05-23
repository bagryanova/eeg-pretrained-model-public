import torch
import torch.nn as nn

from transformers import Wav2Vec2Model

from eeg.models.base import Encoder


class Wav2Vec2Encoder(Encoder):
    def __init__(self, checkpoint):
        super().__init__()

        self._impl = Wav2Vec2Model.from_pretrained(checkpoint)

    def forward(self, tokens):
        x = tokens.transpose(1, 2)

        x, _ = self._impl.feature_projection(x)
        x = self._impl.encoder(x).last_hidden_state
        x = x.transpose(1, 2)

        return {
            'cls': x[:, :, -1],
            'tokens': x,
        }
