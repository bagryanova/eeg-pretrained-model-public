import os
import torch
import torch.nn as nn

from dn3.trainable.layers import ConvEncoderBENDR, BENDRContextualizer
from typing import Dict, Union

from eeg.models.base import Tokenizer, Encoder, Head
from eeg.globals import WEIGHTS_DIR


class BENDRTokenizer(Tokenizer):
    _CHECKPOINTS = {
        'bendr-tokenizer-pretrained': os.path.join(WEIGHTS_DIR, 'bendr_tokenizer.pt'),
    }

    def __init__(self, **kwargs):
        super().__init__()

        self._impl = ConvEncoderBENDR(**kwargs)

    def forward(self, channels):
        return {
            'tokens': self._impl(channels),
        }


class BENDREncoder(Encoder):
    _CHECKPOINTS = {
        'bendr-encoder-pretrained': os.path.join(WEIGHTS_DIR, 'bendr_encoder.pt'),
    }

    def __init__(self, **kwargs):
        super().__init__()

        self._impl = BENDRContextualizer(**kwargs)

    def forward(self, tokens):
        res = self._impl(tokens)
        return {
            'cls': res[:, :, -1],
            'tokens': res[:, :, :-1]
        }
