"""DAGNN. Adapted from https://github.com/dmlc/dgl/blob/master/examples/pytorch/dagnn/main.py.
"""
import argparse
import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
import dgl.function as fn
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops

from sklearn.metrics import accuracy_score as ACC
import copy

class DAGNNConv(nn.Module):
    def __init__(self, in_dim, k):
        super(DAGNNConv, self).__init__()

        self.s = nn.Parameter(torch.FloatTensor(in_dim, 1))
        self.k = k

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("sigmoid")
        nn.init.xavier_uniform_(self.s, gain=gain)

    def forward(self, graph, feats):
        with graph.local_scope():
            results = [feats]

            degs = graph.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm = norm.to(feats.device).unsqueeze(1)

            for _ in range(self.k):
                feats = feats * norm
                graph.ndata["h"] = feats
                graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                feats = graph.ndata["h"]
                feats = feats * norm
                results.append(feats)

            H = torch.stack(results, dim=1)
            S = F.sigmoid(torch.matmul(H, self.s))
            S = S.permute(0, 2, 1)
            H = torch.matmul(S, H).squeeze()

            return H


class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, activation=None, dropout=0):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = 1.0
        if self.activation is F.relu:
            gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, feats):
        feats = self.dropout(feats)
        feats = self.linear(feats)
        if self.activation:
            feats = self.activation(feats)

        return feats

class DAGNN(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        class_num: int,
        device,
        args,
    ):
        super().__init__()
        self.dropout = args.dropout
        self.device = device
        self.lr = args.lr
        self.l2_coef = args.l2_coef
        self.epochs = args.epochs
        self.patience = args.patience

        self.mlp = torch.nn.ModuleList()
        self.mlp.append(
            MLPLayer(
                in_dim=in_features,
                out_dim=args.nhidden,
                bias=True,
                activation=F.relu,
                dropout=self.dropout,
            )
        )
        self.mlp.append(
            MLPLayer(
                in_dim=args.nhidden,
                out_dim=class_num,
                bias=True,
                activation=None,
                dropout=self.dropout,
            )
        )
        self.dagnn = DAGNNConv(in_dim=class_num, k=args.k)

    def forward(self, graph, feats, return_Z=False):
        for layer in self.mlp:
            feats = layer(feats)
        feats = self.dagnn(graph, feats)
        if return_Z:
            return feats, feats
        return feats

    def fit(self, graph, labels, train_mask, val_mask, test_mask):
        # model init
        graph = graph.to(self.device)
        labels = labels.to(self.device)
        self.train_mask = train_mask.to(self.device)
        self.valid_mask = val_mask.to(self.device)
        self.test_mask = test_mask.to(self.device)
        self.to(self.device)

        graph = graph.remove_self_loop().add_self_loop()
        edges = torch.stack(graph.edges()).to(self.device)
        X = graph.ndata["feat"]

        best_epoch = 0
        best_acc = 0.0
        cnt = 0
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_coef)
        loss_fn = torch.nn.CrossEntropyLoss()
        best_state_dict = None

        for epoch in range(self.epochs):
            self.train()
            optimizer.zero_grad()

            Z = self.forward(graph,X)
            loss = loss_fn(Z[self.train_mask], labels[self.train_mask])

            loss.backward()
            optimizer.step()

            [train_acc, valid_acc, test_acc] = self.test(
                graph,
                X,
                labels,
                [self.train_mask, self.valid_mask, self.test_mask],
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
        with torch.no_grad():
            Z = self.forward(graph,X)
            y_pred = torch.argmax(F.softmax(Z, dim=-1), dim=1)
        acc_list = []
        for index in index_list:
            acc_list.append(ACC(labels[index].cpu(), y_pred[index].cpu()))
        return acc_list

    def predict(self, graph):
        self.eval()
        graph = graph.remove_self_loop().add_self_loop()
        graph = graph.to(self.device)
        X = graph.ndata["feat"]
        with torch.no_grad():
            Z, C = self.forward(graph,X, return_Z=True)
            y_pred = torch.argmax(F.softmax(C, dim=-1), dim=1)

        return y_pred.cpu(), C.cpu(), Z.cpu()
