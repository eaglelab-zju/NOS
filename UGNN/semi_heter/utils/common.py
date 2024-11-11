"""
common utils
"""

import numpy as np
import scipy.sparse as sp
import torch


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
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)
