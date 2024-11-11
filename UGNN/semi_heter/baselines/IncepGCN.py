# adapted from :https://github.com/DropEdge/DropEdge/blob/master/src/layers.py

import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch import nn
import torch.nn.functional as F
from ..utils import sys_normalized_adjacency
from sklearn.metrics import accuracy_score as ACC
import copy


class GraphConvolutionBS(Module):
    """
    GCN Layer with BN, Self-loop and Res connection.
    """

    def __init__(
        self,
        in_features,
        out_features,
        activation=lambda x: x,
        withbn=True,
        withloop=True,
        bias=True,
        res=False,
    ):
        """
        Initial function.
        :param in_features: the input feature dimension.
        :param out_features: the output feature dimension.
        :param activation: the activation function.
        :param withbn: using batch normalization.
        :param withloop: using self feature modeling.
        :param bias: enable bias.
        :param res: enable res connections.
        """
        super(GraphConvolutionBS, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = activation
        self.res = res

        # Parameter setting.
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # Is this the best practice or not?
        if withloop:
            self.self_weight = Parameter(torch.FloatTensor(in_features, out_features))
        else:
            self.register_parameter("self_weight", None)

        if withbn:
            self.bn = torch.nn.BatchNorm1d(out_features)
        else:
            self.register_parameter("bn", None)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.self_weight is not None:
            stdv = 1.0 / math.sqrt(self.self_weight.size(1))
            self.self_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)

        # Self-loop
        if self.self_weight is not None:
            output = output + torch.mm(input, self.self_weight)

        if self.bias is not None:
            output = output + self.bias
        # BN
        if self.bn is not None:
            output = self.bn(output)
        # Res
        if self.res:
            return self.sigma(output) + input
        else:
            return self.sigma(output)

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class InecptionGCNBlock(Module):
    """
    The multiple layer GCN with inception connection block.
    """

    def __init__(
        self,
        in_features: int,
        class_num: int,
        device,
        args,
    ):
        super().__init__()
        self.in_features = in_features
        self.hiddendim = args.nhidden
        nbaselayer = args.n_layers
        dropout = args.dropout

        self.device = device
        self.lr = args.lr
        self.l2_coef = args.l2_coef
        self.epochs = args.epochs
        self.patience = args.patience

        withbn = True
        withloop = True
        activation = F.relu
        aggrmethod = "concat"
        self.nbaselayer = nbaselayer
        self.activation = activation
        self.aggrmethod = aggrmethod
        self.dropout = dropout
        self.withbn = withbn
        self.withloop = withloop
        self.midlayers = nn.ModuleList()
        self.__makehidden()

        if self.aggrmethod == "concat":
            self.out_features = in_features + self.hiddendim * nbaselayer
        elif self.aggrmethod == "add":
            if in_features != self.hiddendim:
                raise RuntimeError(
                    "The dimension of in_features and hiddendim should be matched in 'add' model."
                )
            self.out_features = self.hiddendim
        else:
            raise NotImplementedError("The aggregation method only support 'concat', 'add'.")

        self.classifier = nn.Sequential(
            nn.Dropout(0.9),
            torch.nn.Linear(self.out_features, class_num),
        )

    def __makehidden(self):
        # for j in xrange(self.nhiddenlayer):
        for j in range(self.nbaselayer):
            reslayer = nn.ModuleList()
            # for i in xrange(j + 1):
            for i in range(j + 1):
                if i == 0:
                    layer = GraphConvolutionBS(
                        self.in_features,
                        self.hiddendim,
                        self.activation,
                        self.withbn,
                        self.withloop,
                    )
                else:
                    layer = GraphConvolutionBS(
                        self.hiddendim,
                        self.hiddendim,
                        self.activation,
                        self.withbn,
                        self.withloop,
                    )
                reslayer.append(layer)
            self.midlayers.append(reslayer)

    def forward(self, input, adj):
        x = input
        for reslayer in self.midlayers:
            subx = input
            for gc in reslayer:
                subx = gc(subx, adj)
                subx = F.dropout(subx, self.dropout, training=self.training)
            x = self._doconcat(x, subx)
        return self.classifier(x)

    def get_outdim(self):
        return self.out_features

    def _doconcat(self, x, subx):
        if self.aggrmethod == "concat":
            return torch.cat((x, subx), 1)
        elif self.aggrmethod == "add":
            return x + subx

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
        # adj = adj
        self.adj_norm = torch.tensor(
            sys_normalized_adjacency(adj).todense(), device=self.device, dtype=torch.float
        )
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

            Z = self.forward(graph.ndata["feat"], self.adj_norm)
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
            Z = self.forward(graph.ndata["feat"], self.adj_norm)
            y_pred = torch.argmax(Z, dim=1)
        acc_list = []
        for index in index_list:
            acc_list.append(ACC(labels[index].cpu(), y_pred[index].cpu()))
        return acc_list

    def predict(self, graph):
        self.eval()
        graph = graph.to(self.device)
        with torch.no_grad():
            C = self.forward(graph.ndata["feat"], self.adj_norm)
            y_pred = torch.argmax(C, dim=1)

        return y_pred.cpu(), C.cpu(), C.cpu()
