"""DeepSets"""

# pylint: disable=unused-import,line-too-long,unused-argument,too-many-locals
import copy
import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score as ACC
from sklearn.preprocessing import normalize
from the_utils import get_str_time, make_parent_dirs, save_to_csv_files
from torch import nn
from torch.distributions.normal import Normal
from torch.nn import LayerNorm, Linear, Module, ModuleList
from torch.utils.tensorboard import SummaryWriter
from torch_sparse import SparseTensor, fill_diag
from tqdm import tqdm

from .MLP import MLP


class DeepSets(nn.Module):
    def __init__(
        self,
        in_feats,
        h_feats,
        layer_norm: bool = False,
        acts: List[Callable] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hs = [h_feats]
        # n = int(math.log2(in_feats)) - math.log2(h_feats)
        # while n // 2 > 0:
        #     layers += 1
        #     n = n // 2
        #     hs.append(hs[-1] * 2)
        self.phi = MLP(
            in_feats=in_feats,
            h_feats=hs[::-1],
            acts=acts,
            dropout=dropout,
            layer_norm=layer_norm,
        )
        self.rho = MLP(
            in_feats=h_feats,
            h_feats=[h_feats],
            acts=acts,
            dropout=dropout,
            layer_norm=layer_norm,
        )

    def forward(self, x, mean=False):
        if mean:
            h = self.phi(x)
            return self.rho(torch.mean(h, dim=1))
        h = self.phi(x)
        return self.rho(h)
