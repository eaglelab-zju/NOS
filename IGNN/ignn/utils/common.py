"""
common utils
"""

import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import roc_auc_score
from the_utils import make_parent_dirs, split_train_test_nodes
from torch_sparse import SparseTensor
from tqdm import tqdm

flag = True


def metric(
    name,
    logits,
    labels,
    train_mask,
    val_mask,
    test_mask,
):
    if name in [
        "Penn94_linxk",
        "genius_linkx",
        "twitch-gamers_linkx",
        "pokec_linkx",
        "Penn94_linkx",
    ]:
        if labels.min().item() == -1:
            labels[labels == -1] = 0
        if name in ["genius_linkx"]:
            labels = F.one_hot(labels)
    elif name in ["proteins_ogb"]:
        labels = labels.to(torch.float)

    if name not in (
        "yelp-chi_linkx",
        "deezer-europe_linkx",
        # "twitch-gamers_linkx",
        # "pokec_linkx",
        # "Penn94_linkx",
        "fb100_linkx",
        "proteins_ogb",
        "genius_linkx",
    ):
        y_pred = torch.argmax(logits, dim=1)
        return (
            ACC(labels[train_mask].cpu(), y_pred[train_mask].cpu()),
            ACC(labels[val_mask].cpu(), y_pred[val_mask].cpu()),
            ACC(labels[test_mask].cpu(), y_pred[test_mask].cpu()),
        )
    else:
        global flag
        if flag:
            print(f"Evaluate {name} with rocauc.")
            flag = False
        if name in [
            "Penn94_linkx",
            "proteins_ogb",
            "genius_linkx",
            # "twitch-gamers_linkx",
            # "Penn94_linkx",
            "pokec_linkx",
        ]:
            return (
                eval_rocauc(
                    labels[train_mask].cpu().numpy(), logits[train_mask].cpu().numpy()
                )["rocauc"],
                eval_rocauc(
                    labels[val_mask].cpu().numpy(), logits[val_mask].cpu().numpy()
                )["rocauc"],
                eval_rocauc(
                    labels[test_mask].cpu().numpy(), logits[test_mask].cpu().numpy()
                )["rocauc"],
            )


def eval_rocauc(y_true, y_pred):
    """
    compute ROC-AUC and AP score averaged across tasks
    """

    rocauc_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            rocauc_list.append(
                roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
            )

    if len(rocauc_list) == 0:
        raise RuntimeError(
            "No positively labeled data available. Cannot compute ROC-AUC."
        )

    return {"rocauc": sum(rocauc_list) / len(rocauc_list)}


def get_splits_mask(
    name,
    n_nodes,
    train_ratio,
    valid_ratio,
    repeat,
    split_id,
    SPLIT_DIR,
    labeled_idx=None,
):
    if labeled_idx is not None:
        train_idx, val_idx, test_idx = split_train_test_nodes(
            num_nodes=len(labeled_idx),
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            data_name=name,
            split_id=split_id,
            split_times=repeat,
            fixed_split=True,
            split_save_dir=SPLIT_DIR,
        )
        if name != "pokec_linkx":
            train_idx = labeled_idx[train_idx]
            val_idx = labeled_idx[val_idx]
            test_idx = labeled_idx[test_idx]
    else:
        train_idx, val_idx, test_idx = split_train_test_nodes(
            num_nodes=n_nodes,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            data_name=name,
            split_id=split_id,
            split_times=repeat,
            fixed_split=True,
            split_save_dir=SPLIT_DIR,
        )

    train_mask = (
        torch.zeros(n_nodes)
        .scatter_(
            0,
            (
                train_idx.to(torch.int64)
                if isinstance(train_idx, torch.Tensor)
                else torch.LongTensor(train_idx)
            ),
            1,
        )
        .bool()
    )
    val_mask = (
        torch.zeros(n_nodes)
        .scatter_(
            0,
            (
                val_idx.to(torch.int64)
                if isinstance(val_idx, torch.Tensor)
                else torch.LongTensor(val_idx)
            ),
            1,
        )
        .bool()
    )
    test_mask = (
        torch.zeros(n_nodes)
        .scatter_(
            0,
            (
                test_idx.to(torch.int64)
                if isinstance(test_idx, torch.Tensor)
                else torch.LongTensor(test_idx)
            ),
            1,
        )
        .bool()
    )

    return train_mask, val_mask, test_mask


def row_normalized_adjacency(adj, return_deg=False):
    adj = sp.coo_matrix(adj)
    # adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    adj_normalized = adj / row_sum
    if return_deg:
        return sp.coo_matrix(adj_normalized), row_sum
    return sp.coo_matrix(adj_normalized)


def sys_normalized_adjacency(adj, return_deg=False):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    if return_deg:
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo(), rowsum
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.Tensor:
    """Convert a scipy sparse matrix to a torch sparse tensor

    Args:
        sparse_mx (<class 'scipy.sparse'>): sparse matrix

    Returns:
        (torch.Tensor): torch sparse tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


def preprocess_neighborhoods(
    adj: SparseTensor,
    features: torch.FloatTensor,
    name: str,
    n_hops: int,
    set_diag=True,
    remove_diag=False,
    symm_norm=False,
    row_normalized=True,
    device: torch.device = torch.device("cpu"),
    no_save=False,
    return_adj=False,
    process_adj=True,
    save_dir: str = "tmp/flatgnn/neighborhoods",
    batch_idx=None,
    verbose=True,
):
    sym = "sym" if symm_norm else "nsy"
    diag = "diag" if set_diag else "ndiag"
    diag = "ndiag" if remove_diag else diag
    norm = "norm" if row_normalized else "nnorm"
    base = Path(f"{save_dir}/{name}/{norm}/{diag}/{sym}")

    if process_adj:
        if set_diag:
            print("... setting diagonal entries") if verbose else None
            adj = adj.set_diag()
        elif remove_diag:
            print("... removing diagonal entries") if verbose else None
            adj = adj.remove_diag()
        else:
            print("... keeping diag elements as they are") if verbose else None
        if symm_norm:
            print("... performing symmetric normalization") if verbose else None
            deg = adj.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
            adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
        else:
            print("... performing asymmetric normalization") if verbose else None
            deg = adj.sum(dim=1).to(torch.float)
            deg_inv = deg.pow(-1.0)
            deg_inv[deg_inv == float("inf")] = 0
            adj = deg_inv.view(-1, 1) * adj

    # features = F.normalize(features, dim=1) if row_normalized else features
    nei_feats = [
        (
            features.to(device).index_select(0, batch_idx)
            if batch_idx is not None
            else features.to(device)
        )
    ]
    if no_save:
        adj = adj.to_torch_sparse_csr_tensor() if process_adj else adj
        for i in range(1, n_hops + 1):
            x = torch.mm(adj, features.cpu())
            nei_feats.append(x.to(torch.float).to(device))
        if return_adj:
            return nei_feats, adj
        return nei_feats

    adj = adj.to_scipy(layout="csr")
    x = features.cpu().numpy()
    print(f"Load aggregated feats from {base}") if verbose else None
    hops = os.listdir(base) if base.exists() else []
    if f"{n_hops}" not in hops:
        for i in range(1, n_hops + 1):
            file = base.joinpath(f"{i}")
            if file.exists():
                x = torch.load(file, map_location=device)
                x = x.cpu().numpy()
                continue
            make_parent_dirs(file)
            x = adj @ x
            # if name != "arxiv-year_linkx":
            torch.save(
                obj=torch.from_numpy(x).to(torch.float).to(device),
                f=file,
                pickle_protocol=4,
            )

    for i in range(1, n_hops + 1):
        file = base.joinpath(f"{i}")
        x = torch.load(file, map_location=device)
        nei_feats.append(x.index_select(0, batch_idx) if batch_idx is not None else x)

    # for i in range(1, n_hops + 1):
    #     file = base.joinpath(f"{i}")
    #     if file.exists():
    #         x = torch.load(file, map_location=device)
    #         nei_feats.append(x.index_select(0,batch_idx) if batch_idx is not None else x)
    #         continue
    #     x = x.cpu().numpy()
    #     make_parent_dirs(file)
    #     x = adj @ x
    #     x_ = torch.from_numpy(x).to(torch.float).to(device)
    #     nei_feats.append(x_.index_select(0,batch_idx) if batch_idx is not None else x_)
    #     if name!='arxiv-year_linkx':
    #         torch.save(
    #             obj=x_,
    #             f=file,
    #             pickle_protocol=4,
    #         )
    return nei_feats
