from os.path import dirname, abspath, join
import sys

PROJECT_ROOT = dirname(dirname(dirname(dirname(abspath(__file__)))))
sys.path.append(join(PROJECT_ROOT, 'third_party', 'fairseq'))

import torch
import torch.nn as nn
import numpy as np

from typing import List, Tuple
from fairseq.modules import TransposeLast, Fp32LayerNorm, Fp32GroupNorm

from examples.data2vec.models.data2vec2 import Data2VecMultiModel, Data2VecMultiConfig
from examples.data2vec.data.modality import Modality

from eeg.globals import WEIGHTS_DIR
from eeg.models.base import Module


def modify_data2vec_config_default(config):
    config.modalities.audio.decoder.decoder_kernel = 7
    config.modalities.audio.learned_alibi_scale = True
    config.modalities.audio.learned_alibi_scale_per_head = True
    config.modalities.audio.use_alibi_encoder = True
    config.modalities.audio.num_alibi_heads = 12
    config.modalities.audio.decoder.decoder_layers = 4
    config.supported_modality = Modality.AUDIO


class Data2Vec2(Module):
    _CHECKPOINTS = {
        'data2vec2-base-audio': join(WEIGHTS_DIR, 'data2vec2_base_audio.pt'),
        'my-pretrain': join(WEIGHTS_DIR, 'checkpoint_100000.pt'),
    }

    def __init__(self, conv_network=None, modify_config=None):
        super().__init__()

        config = Data2VecMultiConfig()
        if modify_config is not None:
            modify_config(config)

        self._model = Data2VecMultiModel(config, [Modality.AUDIO], skip_ema=False)

        if conv_network is not None:
            self.change_conv_network(conv_network)

    def forward(self, eeg):
        raise NotImplementedError()

    def change_conv_network(self, conv_network):
        self._model.modality_encoders['AUDIO'].local_encoder = conv_network


class Data2Vec2ForPretraining(Data2Vec2):
    def __init__(self, conv_network=None, modify_config=None):
        super().__init__(conv_network=conv_network, modify_config=modify_config)

    def forward(self, channels):
        return self._model(source=channels)

    def set_num_updates(self, step):
        self._model.set_num_updates(step)


class Data2Vec2ForClassification(Data2Vec2):
    def __init__(self, conv_network=None, modify_config=None):
        super().__init__(conv_network=conv_network, modify_config=modify_config)

        self._model.remove_pretraining_modules()

    def forward(self, channels):
        res = {
            'tokens': self._model(source=channels, features_only=True)['x'],
        }
        res['cls'] = res['tokens'][:, 0, :]
        # res['cls'] = torch.mean(res['tokens'], dim=-2)

        return res


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = 'default',
        conv_bias: bool = False,
        in_channels=1,
    ):
        super().__init__()

        assert mode in {'default', 'layer_norm'}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, 'layer norm and group norm are exclusive'

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = in_channels
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, 'invalid conv definition: ' + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == 'layer_norm',
                    is_group_norm=mode == 'default' and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)

        return x
