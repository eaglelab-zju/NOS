"""Preprocess"""

import copy
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from the_utils import make_parent_dirs
from the_utils import split_train_test_nodes


def get_splits_mask(
    graph,
    train_ratio,
    valid_ratio,
    repeat,
    split_id,
    SPLIT_DIR,
):
    train_idx, val_idx, test_idx = split_train_test_nodes(
        num_nodes=graph.num_nodes(),
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        data_name=graph.name,
        split_id=split_id,
        split_times=repeat,
        fixed_split=True,
        split_save_dir=SPLIT_DIR,
    )
    train_mask = (
        torch.zeros(graph.num_nodes()).scatter_(0, torch.tensor(train_idx, dtype=torch.int64),
                                                1).bool()
    )
    val_mask = (
        torch.zeros(graph.num_nodes()).scatter_(0, torch.tensor(val_idx, dtype=torch.int64),
                                                1).bool()
    )
    test_mask = (
        torch.zeros(graph.num_nodes()).scatter_(0, torch.tensor(test_idx, dtype=torch.int64),
                                                1).bool()
    )

    return train_mask, val_mask, test_mask


def split_frequencies(
    adj,
    device: torch.device = torch.device("cpu"),
):
    # TODO: try without symmetric normalization
    I = torch.eye(adj.shape[0]).to(device)
    adj_ = copy.deepcopy(adj + I).to(device)
    _D = torch.diag(adj_.sum(1) * (-0.5))
    tilde_A = _D.matmul(adj_).matmul(_D)

    adj_d = I - tilde_A
    adj_m = tilde_A

    return adj_m.to(device), adj_d.to(device)


def flatten_neighborhood(
    adj: torch.Tensor,
    features: torch.Tensor,
    neighborhood_order: int = 1,
    save_dir: str = None,
    dataset: str = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Flatten the neighborhoods

    Args:
        adj (torch.Tensor): adj without self-loops.
        features (torch.Tensor): features.
        k (int): the order of neighborhood to be flattened.

    Returns:
        torch.Tensor: the list of features of k-th order neighborhood.
    """
    k = neighborhood_order
    save_path = Path(save_dir).joinpath(dataset).joinpath(f"{k}") if save_dir and dataset else None

    if save_path and save_path.exists():
        with open(save_path, "rb") as f:
            fsm, fsd = torch.load(f=f, map_location=device)
        return fsm, fsd

    fsm = [features]
    fsd = [features]
    adj_m, adj_d = split_frequencies(adj)

    for _ in range(k - 1) if k >= 2 else range(k):
        fsm.append(F.normalize(adj_m @ features, dim=1).to(device))
        fsd.append(F.normalize(adj_d @ features, dim=1).to(device))

        adj_m, adj_d = split_frequencies(adj_m)

    fsm, fsd = torch.stack(fsm, dim=0), torch.stack(fsd, dim=0)

    if save_path:
        make_parent_dirs(save_path)
        with open(save_path, "wb") as f:
            torch.save(
                [fsm, fsd],
                f=f,
                pickle_protocol=4,
            )

    return fsm, fsd


def get_neighborhood_features(
    graph,
    adj=None,
    device: torch.device = torch.device("cpu"),
):
    adj = np.array(adj) if adj is not None else graph.adj_external(scipy_fmt="csr").toarray()
    features = graph.ndata["feat"].to(device)

    # adj without self-loops
    adj_nsl = torch.Tensor(adj).to(device)

    fsm, fsd = flatten_neighborhood(
        adj=adj_nsl,
        features=features,
        save_dir="./tmp",
        dataset=graph.name,
    )
    return fsm, fsd, features


def preprocess_graph(
    adj_nsl: sp.csr_matrix,
    layer: int,
    norm: str = "sym",
    renorm: bool = True,
) -> torch.Tensor:
    """Generalized Laplacian Smoothing Filter

    Args:
        adj (sp.csr_matrix): 2D sparse adj without self-loops.
        layer (int):numbers of linear layers
        norm (str):normalize mode of Laplacian matrix
        renorm (bool): If with the renormalization trick

    Returns:
        adjs (sp.csr_matrix):Laplacian Smoothing Filter
    """
    adj = sp.coo_matrix(adj_nsl)
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj

    rowsum = np.array(adj_.sum(1))

    adj_normalized = None
    if norm == "sym":
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized
    elif norm == "left":
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.0).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized
    else:
        raise ValueError("value pf the arg `norm` should be either sym or left")

    reg = [2 / 3] * (layer)

    adjs = []
    for i in reg:
        adjs.append(ident - (i * laplacian))
    return adjs
