"""APPANP. adapted from https://github.com/benedekrozemberczki/APPNP"""

import math
import torch
from torch_sparse import spmm
from scipy import sparse
import numpy as np
from tqdm import trange
import random
import torch.nn.functional as F
import copy
from sklearn.metrics import accuracy_score as ACC
import torch.nn as nn


def normalize_adjacency_matrix(A, I):
    """
    Creating a normalized adjacency matrix with self loops.
    :param A: Sparse adjacency matrix.
    :param I: Identity matrix.
    :return A_tile_hat: Normalized adjacency matrix.
    """
    A_tilde = A + I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sparse.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)
    return A_tilde_hat


def create_propagator_matrix(A, alpha, model):
    """
    Creating  apropagation matrix.
    :param graph: NetworkX graph.
    :param alpha: Teleport parameter.
    :param model: Type of model exact or approximate.
    :return propagator: Propagator matrix Dense torch matrix /
    dict with indices and values for sparse multiplication.
    """
    I = sparse.eye(A.shape[0])
    A_tilde_hat = normalize_adjacency_matrix(A, I)
    if model == "exact":
        propagator = (I - (1 - alpha) * A_tilde_hat).todense()
        propagator = alpha * torch.inverse(torch.FloatTensor(propagator))
    else:
        propagator = dict()
        A_tilde_hat = sparse.coo_matrix(A_tilde_hat)
        indices = np.concatenate(
            [A_tilde_hat.row.reshape(-1, 1), A_tilde_hat.col.reshape(-1, 1)], axis=1
        ).T
        propagator["indices"] = torch.LongTensor(indices)
        propagator["values"] = torch.FloatTensor(A_tilde_hat.data)
    return propagator


def uniform(size, tensor):
    """
    Uniform weight initialization.
    :param size: Size of the tensor.
    :param tensor: Tensor initialized.
    """
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


class DenseFullyConnected(torch.nn.Module):
    """
    Abstract class for PageRank and Approximate PageRank networks.
    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param density: Feature matrix structure.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        uniform(self.out_channels, self.bias)

    def forward(self, features):
        """
        Doing a forward pass.
        :param features: Feature matrix.
        :return filtered_features: Convolved features.
        """
        filtered_features = torch.mm(features, self.weight_matrix)
        filtered_features = filtered_features + self.bias
        return filtered_features


class SparseFullyConnected(torch.nn.Module):
    """
    Abstract class for PageRank and Approximate PageRank networks.
    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param density: Feature matrix structure.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        uniform(self.out_channels, self.bias)

    def forward(self, feature_indices, feature_values):
        """
        Making a forward pass.
        :param feature_indices: Non zero value indices.
        :param feature_values: Matrix values.
        :return filtered_features: Output features.
        """
        number_of_nodes = torch.max(feature_indices[0]).item() + 1
        number_of_features = torch.max(feature_indices[1]).item() + 1
        filtered_features = spmm(
            index=feature_indices,
            value=feature_values,
            m=number_of_nodes,
            n=number_of_features,
            matrix=self.weight_matrix,
        )
        filtered_features = filtered_features + self.bias
        return filtered_features


class APPNPModel(torch.nn.Module):
    """
    APPNP Model Class.
    :param args: Arguments object.
    :param number_of_labels: Number of target labels.
    :param number_of_features: Number of input features.
    :param graph: NetworkX graph.
    :param device: CPU or GPU.
    """

    def __init__(self, args, number_of_labels, number_of_features, graph, device):
        super().__init__()
        self.args = args
        self.number_of_labels = number_of_labels
        self.number_of_features = number_of_features
        self.graph = graph
        self.device = device
        self.setup_layers()
        self.setup_propagator()

    def setup_layers(self):
        """
        Creating layers.
        """
        self.layer_1 = nn.Linear(self.number_of_features, self.args.layers[0])
        self.layer_2 = DenseFullyConnected(self.args.layers[1], self.number_of_labels)

    def setup_propagator(self):
        """
        Defining propagation matrix (Personalized Pagrerank or adjacency).
        """
        self.propagator = create_propagator_matrix(self.graph, self.args.alpha, self.args.type)
        if self.args.type == "exact":
            self.propagator = self.propagator.to(self.device)
        else:
            self.edge_indices = self.propagator["indices"].to(self.device)
            self.edge_weights = self.propagator["values"].to(self.device)

    def forward(self, features):
        """
        Making a forward propagation pass.
        :param feature_indices: Feature indices for feature matrix.
        :param feature_values: Values in the feature matrix.
        :return self.predictions: Predicted class label log softmaxes.
        """

        latent_features_1 = self.layer_1(
            torch.nn.functional.dropout(features, p=self.args.dropout, training=self.training)
        )

        latent_features_1 = torch.nn.functional.relu(latent_features_1)

        latent_features_1 = torch.nn.functional.dropout(
            latent_features_1, p=self.args.dropout, training=self.training
        )

        latent_features_2 = self.layer_2(latent_features_1)
        if self.args.type == "exact":
            self.predictions = torch.nn.functional.dropout(
                self.propagator, p=self.args.dropout, training=self.training
            )

            self.predictions = torch.mm(self.predictions, latent_features_2)
        else:
            localized_predictions = latent_features_2
            edge_weights = torch.nn.functional.dropout(
                self.edge_weights, p=self.args.dropout, training=self.training
            )

            for iteration in range(self.args.iterations):

                new_features = spmm(
                    index=self.edge_indices,
                    value=edge_weights,
                    n=localized_predictions.shape[0],
                    m=localized_predictions.shape[0],
                    matrix=localized_predictions,
                )

                localized_predictions = (1 - self.args.alpha) * new_features
                localized_predictions = localized_predictions + self.args.alpha * latent_features_2
            self.predictions = localized_predictions
        return self.predictions


class APPNP(nn.Module):
    """
    Method to train PPNP/APPNP model.
    """

    def __init__(
        self,
        in_features: int,
        class_num: int,
        device,
        args,
    ) -> None:
        super().__init__()
        # ------------- Parameters ----------------
        self.class_num = class_num
        self.device = device
        self.lr = args.lr
        self.l2_coef = args.l2_coef
        self.epochs = args.epochs
        self.patience = args.patience
        self.n_layers = args.layers
        self.dropout = args.dropout

        self.args = args
        self.graph = args.adj_csr

        self.model = APPNPModel(
            self.args,
            class_num,
            in_features,
            self.graph,
            self.device,
        )

    def fit(self, graph, labels, train_mask, val_mask, test_mask):
        # model init
        graph = graph.to(self.device)
        labels = labels.to(self.device)
        self.train_mask = train_mask.to(self.device)
        self.valid_mask = val_mask.to(self.device)
        self.test_mask = test_mask.to(self.device)
        self.features = graph.ndata["feat"]
        self.to(self.device)

        best_epoch = 0
        best_acc = 0.0
        cnt = 0
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_coef)
        loss_fn = torch.nn.CrossEntropyLoss()
        best_state_dict = None

        for epoch in range(self.epochs):
            self.train()
            optimizer.zero_grad()

            Z = self.forward(self.features)
            loss = loss_fn(Z[self.train_mask], labels[self.train_mask])

            loss.backward()
            optimizer.step()

            [train_acc, valid_acc, test_acc] = self.test(
                graph, self.features, labels, [self.train_mask, self.valid_mask, self.test_mask]
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

    def forward(self, features, return_Z=False):
        Z = self.model(features)
        C = Z
        if return_Z:
            return Z, C
        return C

    def test(self, graph, X, labels, index_list):
        self.eval()
        with torch.no_grad():
            Z = self.forward(X)
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
            Z, C = self.forward(X, return_Z=True)
            y_pred = torch.argmax(F.softmax(Z, dim=-1), dim=1)

        return y_pred.cpu(), C.cpu(), Z.cpu()
