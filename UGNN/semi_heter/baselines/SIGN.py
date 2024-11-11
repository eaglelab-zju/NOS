# Semi-supervised Classification with Graph Convolutional Networks.
# code adapted from :https://github.com/dmlc/dgl/blob/master/examples/pytorch/sign/sign.py
import copy
import math
import os
import random
import time

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv
from sklearn.metrics import accuracy_score as ACC
from torch.nn import init
import argparse
import os
import time

import dgl
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.LayerNorm(hidden))
                self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
            self.prelu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                x = self.dropout(self.prelu(x))
        return x


class SIGN(nn.Module):
    def __init__(
        self,
        in_features: int,
        class_num: int,
        device,
        args,
    ):
        super().__init__()
        in_feats = in_features
        hidden = args.nhidden
        out_feats = class_num
        self.hops = args.n_hops
        n_layers = args.n_layers
        dropout = args.dropout
        self.lr = args.lr
        self.l2_coef = args.l2_coef
        self.epochs = args.epochs
        self.patience = args.patience

        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.ReLU()
        self.inception_ffs = nn.ModuleList()
        for hop in range(self.hops + 1):
            self.inception_ffs.append(FeedForwardNet(in_feats, hidden, hidden, n_layers, dropout))
        # self.linear = nn.Linear(hidden * (hops+ 1), out_feats)
        self.project = FeedForwardNet((self.hops + 1) * hidden, hidden, out_feats, 1, dropout=0.9)

    def forward(self, feats):
        hidden = []
        for feat, ff in zip(feats, self.inception_ffs):
            hidden.append(ff(feat))
        hidden = torch.cat(hidden, dim=-1)
        hidden = F.layer_norm(hidden, [hidden.shape[1]])
        out = self.project(self.dropout(self.prelu(hidden)))
        return out

    def fit(self, graph, labels, train_mask, val_mask, test_mask):
        # model init
        graph = graph.to(self.device)
        labels = labels.to(self.device)
        self.train_mask = train_mask.to(self.device)
        self.valid_mask = val_mask.to(self.device)
        self.test_mask = test_mask.to(self.device)
        self.to(self.device)

        graph = graph.remove_self_loop().add_self_loop()
        adj = graph.adj_external(scipy_fmt="csr")
        adj = torch.tensor(adj.todense(), device=self.device, dtype=torch.float)
        X = graph.ndata["feat"]
        n_nodes, _ = X.shape

        best_epoch = 0
        best_acc = 0.0
        cnt = 0
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_coef)
        loss_fn = torch.nn.CrossEntropyLoss()
        best_state_dict = None
        self.feats = preprocess(graph, graph.ndata["feat"], self.hops)
        for epoch in range(self.epochs):
            self.train()

            Z = self.forward(self.feats)
            loss = loss_fn(Z[self.train_mask], labels[self.train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            [train_acc, valid_acc, test_acc] = self.test(
                graph, X, labels, [self.train_mask, self.valid_mask, self.test_mask]
            )

            if valid_acc > best_acc:
                cnt = 0
                best_acc = valid_acc
                best_epoch = epoch
                best_state_dict = copy.deepcopy(self.state_dict())
                print(f"\nEpoch:{epoch}, Loss:{loss.item()}")
                print(
                    f"train acc: {train_acc:.3f} valid acc: {valid_acc:.3f}, test acc: {test_acc:.3f}"
                )

            else:
                cnt += 1
                if cnt == self.patience:
                    print(f"Early Stopping! Best Epoch: {best_epoch}, best val acc: {best_acc}")
                    break

        self.load_state_dict(best_state_dict)
        self.best_epoch = best_epoch

    def test(self, graph, X, labels, index_list):
        self.eval()
        self.train(False)
        with torch.no_grad():
            Z = self.forward(self.feats)
            y_pred = torch.argmax(Z, dim=1)
        acc_list = []
        for index in index_list:
            acc_list.append(ACC(labels[index].cpu(), y_pred[index].cpu()))
        return acc_list

    def predict(self, graph):
        self.eval()
        self.train(False)
        graph = graph.remove_self_loop().add_self_loop()
        graph = graph.to(self.device)
        X = graph.ndata["feat"]
        with torch.no_grad():
            C = self.forward(self.feats)
            y_pred = torch.argmax(C, dim=1)

        return y_pred.cpu(), C.cpu(), C.cpu()


def calc_weight(g):
    """
    Compute row_normalized(D^(-1/2)AD^(-1/2))
    """
    with g.local_scope():
        # compute D^(-0.5)*D(-1/2), assuming A is Identity
        g.ndata["in_deg"] = g.in_degrees().float().pow(-0.5)
        g.ndata["out_deg"] = g.out_degrees().float().pow(-0.5)
        g.apply_edges(fn.u_mul_v("out_deg", "in_deg", "weight"))
        # row-normalize weight
        g.update_all(fn.copy_e("weight", "msg"), fn.sum("msg", "norm"))
        g.apply_edges(fn.e_div_v("weight", "norm", "weight"))
        return g.edata["weight"]


def preprocess(g, features, R):
    """
    Pre-compute the average of n-th hop neighbors
    """
    with torch.no_grad():
        g.edata["weight"] = calc_weight(g)
        g.ndata["feat_0"] = features
        for hop in range(1, R + 1):
            g.update_all(
                fn.u_mul_e(f"feat_{hop-1}", "weight", "msg"),
                fn.sum("msg", f"feat_{hop}"),
            )
        res = []
        for hop in range(R + 1):
            res.append(g.ndata.pop(f"feat_{hop}"))
        return res
