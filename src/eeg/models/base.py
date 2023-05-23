import torch
import torch.nn as nn

from typing import Dict


class Module(nn.Module):
    def load_checkpoint(self, name):
        if name in self._CHECKPOINTS:
            path = self._CHECKPOINTS[name]
        else:
            path = name

        try:
            self.load_state_dict(torch.load(path, map_location='cpu'))
        except:
            self.load_state_dict(torch.load(path, map_location='cpu')['model_state_dict'])
        return self


class Tokenizer(Module):
    def forward(self, **kwargs) -> Dict:
        raise NotImplementedError()


class Encoder(Module):
    def forward(self, **kwargs) -> Dict:
        raise NotImplementedError()


class Head(Module):
    def forward(self, **kwargs) -> Dict:
        raise NotImplementedError()
