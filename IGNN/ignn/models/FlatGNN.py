"""FlatGNN"""

# pylint: disable=unused-import,line-too-long,unused-argument,too-many-locals
import copy
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from the_utils import get_str_time, make_parent_dirs, save_to_csv_files
from torch import nn
from torch.distributions.normal import Normal
from torch.nn import LayerNorm, Linear, Module, ModuleList
from torch.utils.tensorboard import SummaryWriter
from torch_sparse import SparseTensor, fill_diag
from tqdm import tqdm

from ..modules import MLP
from ..modules import FlatGNN as FlatGNN_layer
from ..utils import eval_rocauc, metric

losses = {
    "ce": torch.nn.CrossEntropyLoss,
    "bce": torch.nn.BCEWithLogitsLoss,
}


class FlatGNN(nn.Module):
    """FlatGNN"""

    def __init__(
        self,
        in_feats,
        h_feats,
        n_clusters,
        n_epochs=2000,
        lr: float = 0.001,
        l2_coef: float = 0.00005,
        early_stop: int = 100,
        device=None,
        nas_dropout: float = 0.0,
        nss_dropout: float = 0.8,
        clf_dropout: float = 0.9,
        out_ndim_trans: int = 64,
        lda: float = 1,
        n_hops=6,
        n_intervals=3,
        nie="gcn",
        nrl="concat",
        n_layers=1,
        act="relu",
        layer_norm=True,
        loss="ce",
        n_nodes=None,
        ndim_h_a=64,
        num_heads=1,
        transform_first=False,
        trans_layer_num=5,
        no_save=True,
    ) -> None:
        super().__init__()

        assert loss in losses.keys(), f"loss should be in {losses.keys()}"

        self.n_intervals = n_intervals
        self.nss_dropout = nss_dropout

        # FlatGNN
        self.n_epochs = n_epochs
        self.h_feats = h_feats
        self.l2_coef = l2_coef
        self.n_nodes = n_nodes

        # URL
        self.lr = lr
        self.estop_steps = early_stop
        self.n_clusters = n_clusters
        self.device = device
        self.best_model = None

        self.flat_gnn = FlatGNN_layer(
            in_feats=in_feats,
            h_feats=h_feats,
            n_hops=n_hops,
            nas_dropout=nas_dropout,
            nss_dropout=nss_dropout,
            n_intervals=n_intervals,
            out_ndim_trans=out_ndim_trans,
            nie=nie,
            nrl=nrl,
            act=act,
            layer_norm=layer_norm,
            n_nodes=n_nodes,
            ndim_h_a=ndim_h_a,
            num_heads=num_heads,
            transform_first=transform_first,
            trans_layer_num=trans_layer_num,
            ignn_layer_num=n_layers,
            no_save=no_save,
        )

        hidden_dim = {
            "multi-con": max(h_feats * (n_hops - n_intervals + 2), h_feats),
            "concat": h_feats + ndim_h_a if n_nodes is not None else h_feats,
            # "concat": h_feats,
            "ordered-gating": h_feats,
            # "self-attention": h_feats + out_ndim_trans * num_heads + ndim_h_a if n_nodes is not None else h_feats + out_ndim_trans * num_heads,
            # "self-attention":h_feats,
            "self-attention": out_ndim_trans * num_heads if trans_layer_num else h_feats,
            # "self-attention":out_ndim_trans * num_heads + ndim_h_a if n_nodes is not None else out_ndim_trans * num_heads,
            # "self-attention": (
            #     (
            #         out_ndim_trans * num_heads + h_feats + ndim_h_a
            #         if n_nodes is not None
            #         else out_ndim_trans * num_heads + h_feats
            #     )
            #     if trans_layer_num != 0
            #     else h_feats + ndim_h_a if n_nodes is not None else h_feats
            # ),
            # "self-attention": out_ndim_trans * num_heads,
            "only-concat": h_feats * (n_hops + 1),
            "max": h_feats,
            "mean": h_feats,
            "sum": h_feats,
            "lstm": h_feats,
            "none": h_feats,
        }[nrl]

        # self.flat_gnn_s = None
        # if n_layers > 1:
        #     self.beta = lda / n_layers
        #     self.flat_gnn_s = nn.ModuleList(
        #         FlatGNN_layer(
        #             in_feats=hidden_dim,
        #             h_feats=h_feats,
        #             n_hops=n_hops,
        #             nas_dropout=nas_dropout,
        #             nss_dropout=nss_dropout,
        #             n_intervals=n_intervals,
        #             out_ndim_trans=out_ndim_trans,
        #             no_save=True,
        #             nie=nie,
        #             nrl=nrl,
        #             act=act,
        #             layer_norm=layer_norm,
        #             n_nodes=n_nodes,
        #             ndim_h_a=ndim_h_a,
        #             num_heads=num_heads,
        #             transform_first=transform_first,
        #             trans_layer_num=trans_layer_num,
        #         )
        #         for _ in range(n_layers - 1)
        #     )

        self.classifier = MLP(
            in_feats=hidden_dim,
            h_feats=[n_clusters],
            acts=[nn.Identity()],
            dropout=clf_dropout,
            layer_norm=False,
        )

        self.criterion = losses[loss]()

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.l2_coef,
        )

    def forward(
        self,
        features,
        batch_idx=None,
        graph=None,
        device=None,
    ):
        z_flat_gnn = self.flat_gnn(graph=graph, device=device, batch_idx=batch_idx, feats=features)
        # if self.flat_gnn_s is not None:
        #     for _, flat_gnn_s in enumerate(self.flat_gnn_s):
        #         # z_flat_gnn = (
        #         #     self.beta * flat_gnn_s(graph=graph, device=device, feats=z_flat_gnn)
        #         #     + (1 - self.beta) * z_flat_gnn
        #         # )
        #         z_flat_gnn = flat_gnn_s(graph=graph, device=device, feats=z_flat_gnn)

        return z_flat_gnn

    def validate(
        self,
        features,
        labels,
        train_mask,
        val_mask,
        test_mask,
        graph,
        batch_idx=None,
    ):
        self.train(False)
        with torch.no_grad():
            # H,embeddings = self.forward(
            #     features,
            #     graph=graph,
            #     device=self.device,
            #     batch_idx=batch_idx,
            # )
            embeddings = self.forward(
                features,
                graph=graph,
                device=self.device,
                batch_idx=batch_idx,
            )
            logits = self.classifier(embeddings)

            return metric(
                graph.name,
                logits,
                labels,
                train_mask,
                val_mask,
                test_mask,
            )

    def fit(
        self,
        graph,
        labels,
        train_mask,
        val_mask,
        test_mask,
        bs=None,
        split_id=None,
        device: torch.device = torch.device("cpu"),
    ):

        self.device = device
        self.to(self.device)

        if graph.name in ["proteins_ogb"]:
            labels = labels.to(torch.float).to(device)
        else:
            labels = labels.to(device)

        best_epoch = 0
        best_acc = 0.0
        cnt = 0
        best_state_dict = None
        # writer = SummaryWriter(
        #     log_dir=f"logs/runs/{get_str_time()[:10]}/joint_{graph.name}_{split_id}_{self.h_feats}_{get_str_time()[11:]}"
        # )

        t_start = time.time()
        for epoch in range(self.n_epochs):

            n = graph.num_nodes()
            shf = torch.randperm(n)

            loss_value = 0
            loss_val_value = 0
            loss_test_value = 0

            if bs is not None:
                pred_stack = []
                for step in tqdm(range(0, n, bs), "training step"):
                    batch_idx = shf[step : step + bs]

                    batch_labels = labels[batch_idx]
                    batch_train_mask = train_mask[batch_idx]
                    batch_val_mask = val_mask[batch_idx]
                    batch_test_mask = test_mask[batch_idx]

                    self.train()
                    self.optimizer.zero_grad()

                    embeddings = self.forward(
                        graph.ndata["feat"],
                        batch_idx=batch_idx,
                        graph=graph,
                        device=device,
                    )
                    # H,embeddings = self.forward(
                    #     graph.ndata["feat"],
                    #     batch_idx=batch_idx,
                    #     graph=graph,
                    #     device=device,
                    # )

                    logits = self.classifier(embeddings)
                    loss = self.criterion(logits[batch_train_mask], batch_labels[batch_train_mask])
                    # logits_1 = self.classifier_1(H)
                    # loss = self.criterion(logits[batch_train_mask], batch_labels[batch_train_mask]) + self.criterion(logits_1[batch_train_mask], batch_labels[batch_train_mask])
                    loss_val = self.criterion(logits[batch_val_mask], batch_labels[batch_val_mask])
                    loss_test = self.criterion(
                        logits[batch_test_mask], batch_labels[batch_test_mask]
                    )

                    loss.backward()
                    self.optimizer.step()

                    loss_value += loss.item()
                    loss_val_value += loss_val.item()
                    loss_test_value += loss_test.item()

                    pred_stack.append(logits.detach().clone().cpu())

                if epoch % 9 == 0:
                    with torch.no_grad():
                        self.train(False)
                        pred_stack = []
                        idx = torch.LongTensor(list(range(n)))
                        for step in tqdm(range(0, n, bs), "validate step"):
                            batch_idx = idx[step : step + bs]

                            embeddings = self.forward(
                                graph.ndata["feat"],
                                batch_idx=batch_idx,
                                graph=graph,
                                device=device,
                            )
                            # H, embeddings = self.forward(
                            #     graph.ndata["feat"],
                            #     batch_idx=batch_idx,
                            #     graph=graph,
                            #     device=device,
                            # )
                            logits = self.classifier(embeddings)
                            pred_stack.append(logits)

                        pred_stack = torch.cat(pred_stack, dim=0)

                        train_acc, valid_acc, test_acc = metric(
                            graph.name,
                            logits=pred_stack,
                            labels=labels,
                            train_mask=train_mask,
                            val_mask=val_mask,
                            test_mask=test_mask,
                        )
                else:
                    pred_stack = torch.cat(pred_stack, dim=0)

                    train_acc, valid_acc, test_acc = metric(
                        graph.name,
                        logits=pred_stack,
                        labels=labels[shf],
                        train_mask=train_mask[shf],
                        val_mask=val_mask[shf],
                        test_mask=test_mask[shf],
                    )

            else:
                self.train()
                self.optimizer.zero_grad()

                embeddings = self.forward(
                    graph.ndata["feat"],
                    batch_idx=None,
                    graph=graph,
                    device=device,
                )
                # H, embeddings = self.forward(
                #     graph.ndata["feat"],
                #     batch_idx=None,
                #     graph=graph,
                #     device=device,
                # )

                logits = self.classifier(embeddings)
                loss = self.criterion(logits[train_mask], labels[train_mask])
                # logits_1 = self.classifier_1(H)
                # loss = self.criterion(logits[train_mask], labels[train_mask]) + self.criterion(logits_1[train_mask], labels[train_mask])
                # loss = self.criterion(logits[train_mask], labels[train_mask])
                loss_val = self.criterion(logits[val_mask], labels[val_mask])
                loss_test = self.criterion(logits[test_mask], labels[test_mask])

                loss.backward()
                self.optimizer.step()

                loss_value = loss.item()

                (train_acc, valid_acc, test_acc) = self.validate(
                    graph.ndata["feat"],
                    labels,
                    train_mask,
                    val_mask,
                    test_mask,
                    graph=graph,
                    batch_idx=None,
                )

            print(
                f"epoch:{epoch},loss: {loss_value}, train acc: {train_acc:.3f}, valid acc: {valid_acc:.3f}, test acc: {test_acc:.3f}"
            )
            if valid_acc >= best_acc:
                cnt = 0
                best_acc = valid_acc
                best_acc_t = test_acc
                best_epoch = epoch
                best_state_dict = copy.deepcopy(self.state_dict())
                # print(f"\nEpoch:{epoch}, Loss:{loss.item()}")
                # print(
                #     f"train acc: {train_acc:.3f}, valid acc: {valid_acc:.3f}, test acc: {test_acc:.3f}"
                # )
            else:
                cnt += 1
                if cnt == self.estop_steps:
                    print(
                        f"{graph.name} Early Stopping! Best Epoch: {best_epoch}, best val acc: {best_acc:.3f}, test acc: {best_acc_t:.3f}"
                    )
                    break

            # writer.add_scalar("joint/time/train", time.time() - t, epoch)

            # writer.add_scalar("joint/metric/train", train_acc, epoch)
            # writer.add_scalar("joint/metric/val", valid_acc, epoch)
            # writer.add_scalar("joint/metric/test", test_acc, epoch)
            # writer.add_scalar("joint/loss/train", loss.item(), epoch)
            # writer.add_scalar("joint/loss/val", loss_val.item(), epoch)
            # writer.add_scalar("joint/loss/test", loss_test.item(), epoch)

        t_finish = time.time()
        print(f"10 epoch cost: {(t_finish-t_start)/epoch * 10:.4f}s")
        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)

    def get_embeddings(self):
        with torch.no_grad():
            pass
