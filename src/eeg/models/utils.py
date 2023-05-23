import torch
import torch.nn as nn

from typing import Dict, Union

from eeg.models.base import Module, Tokenizer, Encoder, Head


class GenericClassifier(nn.Module):
    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        encoder: Encoder,
        heads: Union[Head, Dict[str, Head]]):

        super().__init__()

        self._tokenizer = tokenizer
        self._encoder = encoder
        self._heads = heads

    def forward(self, **kwargs):
        x = self._tokenizer(**kwargs)
        if self._encoder is not None:
            x = self._encoder(**x)
        if isinstance(self._heads, nn.Module):
            return self._heads(**x)

        res = dict()
        for name, h in self._heads.items():
            res[name] = h(**x)
        return res


class ApplyTo(nn.Module):
    def __init__(self, in_name, out_name, module):
        super().__init__()

        self._in_name = in_name
        self._out_name = out_name
        self._module = module

    def forward(self, **kwargs):
        res = kwargs
        res[self._out_name] = self._module(kwargs[self._in_name])
        return res


class LinearHead(Head):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self._model = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self._model(x)


class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        self._modules = args

    def forward(self, **kwargs):
        x = kwargs
        for m in self._modules:
            x = m(**x)
        return x


class AddCLSToken(nn.Module):
    def __init__(self, submodule, dim, std=0.01):
        super().__init__()

        self._submodule = submodule
        self._token = nn.Parameter(torch.zeros((1, 1, dim)), requires_grad=True)
        self._token.data.normal_(mean=0.0, std=std)

    def forward(self, x, *args, **kwargs):
        x = self._submodule(x, *args, **kwargs)
        x = torch.cat([self._token.expand(x.shape[0], 1, x.shape[2]), x], dim=-2)
        return x


class IgnoreFirstToken(nn.Module):
    def __init__(self, submodule):
        super().__init__()

        self._submodule = submodule

    def forward(self, x, *args, **kwargs):
        x = self._submodule(x[:, 1:, :], *args, **kwargs)
        z = torch.zeros((x.shape[0], 1, x.shape[2]), device=x.device)
        x = torch.cat([z, x], dim=-2)
        return x
