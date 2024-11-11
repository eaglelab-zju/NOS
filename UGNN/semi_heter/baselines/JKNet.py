# code adapted from :https://github.com/mori97/JKNet-dgl
import torch
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
from tqdm import tqdm
import copy
from sklearn.metrics import accuracy_score as ACC
from dgl.nn.pytorch import GraphConv


class GraphConvLayer(torch.nn.Module):
    """Graph convolution layer.

    Args:
        in_features (int): Size of each input node.
        out_features (int): Size of each output node.
        aggregation (str): 'sum', 'mean' or 'max'.
                           Specify the way to aggregate the neighbourhoods.
    """

    AGGREGATIONS = {
        "sum": torch.sum,
        "mean": torch.mean,
        "max": torch.max,
    }

    def __init__(self, in_features, out_features, aggregation="sum"):
        super(GraphConvLayer, self).__init__()

        if aggregation not in self.AGGREGATIONS.keys():
            raise ValueError("'aggregation' argument has to be one of " "'sum', 'mean' or 'max'.")
        self.aggregate = lambda nodes: self.AGGREGATIONS[aggregation](nodes, dim=1)

        self.linear = torch.nn.Linear(in_features, out_features)
        self.self_loop_w = torch.nn.Linear(in_features, out_features)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, graph, x):
        graph.ndata["h"] = x
        graph.update_all(
            fn.copy_u(u="h", out="msg"),
            lambda nodes: {"h": self.aggregate(nodes.mailbox["msg"])},
        )
        h = graph.ndata.pop("h")
        h = self.linear(h)
        return h + self.self_loop_w(x) + self.bias


class JKNetConcat(torch.nn.Module):
    """An implementation of Jumping Knowledge Network (arxiv 1806.03536) which
    combine layers with concatenation.

    Args:
        in_features (int): Size of each input node.
        out_features (int): Size of each output node.
        n_layers (int): Number of the convolution layers.
        n_units (int): Size of the middle layers.
        aggregation (str): 'sum', 'mean' or 'max'.
                           Specify the way to aggregate the neighbourhoods.
    """

    def __init__(self,
        in_features: int,
        class_num: int,
        device,
        args,
    ):
        super(JKNetConcat, self).__init__()
        n_units = args.n_units
        aggregation = args.aggregation
        self.n_layers = args.n_layers
        self.lr = args.lr
        self.l2_coef = args.l2_coef
        self.epochs = args.epochs
        self.patience = args.patience
        self.dropout = args.dropout
        self.device = device

        # self.gconv0 = GraphConvLayer(in_features, n_units, aggregation)
        self.gconv0 = GraphConv(in_features, n_units)
        self.dropout0 = torch.nn.Dropout(self.dropout)
        for i in range(1, self.n_layers):
            # setattr(self, "gconv{}".format(i), GraphConvLayer(n_units, n_units, aggregation))
            setattr(self, "gconv{}".format(i), GraphConv(n_units, n_units))
            setattr(self, "dropout{}".format(i), torch.nn.Dropout(self.dropout))
        self.last_linear = torch.nn.Linear(self.n_layers * n_units, class_num)

    def forward(self, graph, x):
        layer_outputs = []
        for i in range(self.n_layers):
            dropout = getattr(self, "dropout{}".format(i))
            gconv = getattr(self, "gconv{}".format(i))
            x = dropout(F.relu(gconv(graph, x)))
            layer_outputs.append(x)

        h = torch.cat(layer_outputs, dim=1)
        return self.last_linear(h)

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

        for epoch in range(self.epochs):
            self.train()

            Z = self.forward(graph, X)
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
        with torch.no_grad():
            Z = self.forward(graph, X)
            y_pred = torch.argmax(Z, dim=1)
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
            C = self.forward(graph, X)
            y_pred = torch.argmax(C, dim=1)

        return y_pred.cpu(), C.cpu(), C.cpu()


class JKNetMaxpool(torch.nn.Module):
    """An implementation of Jumping Knowledge Network (arxiv 1806.03536) which
    combine layers with Maxpool.

    Args:
        in_features (int): Size of each input node.
        out_features (int): Size of each output node.
        n_layers (int): Number of the convolution layers.
        n_units (int): Size of the middle layers.
        aggregation (str): 'sum', 'mean' or 'max'.
                           Specify the way to aggregate the neighbourhoods.
    """

    def __init__(
        self,
        in_features: int,
        class_num: int,
        device,
        args,
    ):
        super(JKNetMaxpool, self).__init__()
        n_units = args.n_units
        aggregation = args.aggregation
        self.n_layers = args.n_layers
        self.lr = args.lr
        self.l2_coef = args.l2_coef
        self.epochs = args.epochs
        self.patience = args.patience
        self.dropout = args.dropout
        self.device = device

        # self.gconv0 = GraphConvLayer(in_features, n_units, aggregation)
        self.gconv0 = GraphConv(in_features, n_units)
        self.dropout0 = torch.nn.Dropout(self.dropout)
        for i in range(1, self.n_layers):
            # setattr(self, "gconv{}".format(i), GraphConvLayer(n_units, n_units, aggregation))
            setattr(self, "gconv{}".format(i), GraphConv(n_units, n_units))
            setattr(self, "dropout{}".format(i), torch.nn.Dropout(self.dropout))
        self.last_linear = torch.nn.Linear(n_units, class_num)

    def forward(self, graph, x):
        layer_outputs = []
        for i in range(self.n_layers):
            dropout = getattr(self, "dropout{}".format(i))
            gconv = getattr(self, "gconv{}".format(i))
            x = dropout(F.relu(gconv(graph, x)))
            layer_outputs.append(x)

        h = torch.stack(layer_outputs, dim=0)
        h = torch.max(h, dim=0)[0]
        return self.last_linear(h)

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

        for epoch in range(self.epochs):
            self.train()

            Z = self.forward(graph, X)
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
        with torch.no_grad():
            Z = self.forward(graph, X)
            y_pred = torch.argmax(Z, dim=1)
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
            C = self.forward(graph, X)
            y_pred = torch.argmax(C, dim=1)

        return y_pred.cpu(), C.cpu(), C.cpu()
