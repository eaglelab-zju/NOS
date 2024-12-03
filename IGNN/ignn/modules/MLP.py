"""MLP"""

# pylint: disable=unused-import,line-too-long,unused-argument,too-many-locals
from typing import Any, Callable, Dict, List, Tuple

import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """MLP"""

    def __init__(
        self,
        in_feats: int,
        h_feats: List[int],
        layer_norm: bool = False,
        acts: List[Callable] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = len(h_feats)
        module_list = []
        in_feats = [in_feats] + h_feats
        for i in range(self.layers):
            if acts is not None and len(acts) > 0 and acts[i] is not None:
                module_list.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(in_feats[i], h_feats[i]),
                        nn.LayerNorm(h_feats[i]),
                        acts[i],
                    )
                    if layer_norm
                    else nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(in_feats[i], h_feats[i]),
                        acts[i],
                    )
                )
            else:
                module_list.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(in_feats[i], h_feats[i]),
                        nn.LayerNorm(h_feats[i]),
                    )
                    if layer_norm
                    else nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(in_feats[i], h_feats[i]),
                        acts[i],
                    )
                )
        self.ec = nn.ModuleList(module_list)

    def forward(self, x):
        z = x
        for i in range(self.layers):
            z = self.ec[i](z)
        return z
