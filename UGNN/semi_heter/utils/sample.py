"""Sample utils"""

import random
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import scipy.sparse as sp
import torch


def get_batch_edges(
    adj_csr: sp.csr_matrix,
    nodes_batch: np.array,
    sample_neg: bool = True,
) -> Union[Tuple[Tuple[np.array]], Tuple[Tuple[Tuple[np.array]]]]:
    nodes_batch_set, _ = torch.unique(nodes_batch, return_inverse=True)
    submatrix = adj_csr[nodes_batch_set, :][:, nodes_batch_set]
    rows, cols = submatrix.nonzero()
    pos_edges = (nodes_batch_set[rows], nodes_batch_set[cols])
    if sample_neg:
        (u, v), (neg_rows, neg_cols) = sample_neg_edges(submatrix, len(rows))
        return (pos_edges, (rows, cols)), (
            (nodes_batch_set[u], nodes_batch_set[v]),
            (neg_rows, neg_cols),
        )
    return (pos_edges, (rows, cols))


def sample_neg_edges(
    submatrix: sp.csr_matrix,
    num_samples: int,
) -> List[Tuple[int]]:
    rows, cols = np.where(submatrix.A == 0)
    sampled_idx = random.sample(range(len(rows)), k=num_samples)
    return ((rows[sampled_idx], cols[sampled_idx]), (rows[sampled_idx], cols[sampled_idx]))
