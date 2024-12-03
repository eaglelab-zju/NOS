import json
import random

import cv2
import dgl
import networkx as nx
import numpy as np
import torch
import torchvision.transforms as transforms
from dgl.data import DGLDataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from utils import is_focusable, is_valid


class GNNDataset(DGLDataset):
    def __init__(self, config, mode="train"):
        self.config = config
        self.ids = (
            self.config.dataset_config.train_data_ids
            if mode == "train"
            else self.config.dataset_config.valid_data_ids
        )
        self.mode = mode

        super().__init__(name="GNNDataset")

    def process(self):
        (
            self.page_images,
            self.page_texts,
            self.page_attributes,
            self.page_coordinates_normal,
            self.page_masks,
            self.page_labels,
            self.graphs,
        ) = ([], [], [], [], [], [], [])
        for page_id in tqdm(self.ids):
            with open(
                f"{self.config.dataset_config.hierarchy_dir}/{page_id}.json",
                "r",
                encoding="utf-8",
            ) as f:
                nodes = json.load(f)["nodes"]

            (
                node_images,
                node_texts,
                node_attributes,
                node_coordinates_normal,
                node_masks,
                node_labels,
            ) = ([], [], [], [], [], [])
            for node in nodes:
                node_masks.append(int(is_valid(node)))
                node_texts.append(
                    (
                        f"{self.config.dataset_config.text_feat_dir}/{page_id}.pt",
                        node["id"],
                    )
                )
                # 添加 clickable 噪声
                if (
                    self.mode == "train"
                    and node["attributes"][5] == 1
                    and random.random()
                    <= self.config.dataset_config.clickable_noise_ratio
                ):
                    node["attributes"][5] = 0
                node_attributes.append(node["attributes"])
                node_coordinates_normal.append(node["coordinate_normal"])
                node_labels.append(int(is_focusable(node)))
                node_images.append(
                    (
                        f"{self.config.dataset_config.box_feat_dir}/{page_id}.pt",
                        node["id"],
                    )
                )

            self.page_images.append(node_images)
            self.page_texts.append(node_texts)
            self.page_attributes.append(node_attributes)
            self.page_coordinates_normal.append(node_coordinates_normal)
            self.page_masks.append(node_masks)
            self.page_labels.append(node_labels)
            self.graphs.append(f"{self.config.dataset_config.graph_dir}/{page_id}.npy")

    def __getitem__(self, idx):
        (
            node_images,
            node_texts,
            node_attributes,
            node_coordinates_normal,
            node_masks,
            node_labels,
            graph,
        ) = (
            self.page_images[idx],
            self.page_texts[idx],
            self.page_attributes[idx],
            self.page_coordinates_normal[idx],
            self.page_masks[idx],
            self.page_labels[idx],
            self.graphs[idx],
        )

        page_text_feats = torch.load(
            node_texts[0][0], map_location="cpu", weights_only=True
        )
        page_image_feats = torch.load(
            node_images[0][0], map_location="cpu", weights_only=True
        )
        images, texts = [], []
        for (_, node_id), (_, node_id) in zip(node_images, node_texts):
            texts.append(page_text_feats[node_id])
            images.append(page_image_feats[node_id])

        graph = np.load(graph)
        graph = nx.from_numpy_array(graph, create_using=nx.DiGraph).to_directed()
        graph = dgl.from_networkx(graph)
        graph = dgl.add_self_loop(graph)

        graph.ndata["images"] = torch.stack(images, dim=0)  # N * 1000
        graph.ndata["texts"] = torch.stack(texts, dim=0)  # N * (768 * 4)
        graph.ndata["attributes"] = torch.FloatTensor(node_attributes)  # N * 11
        graph.ndata["labels"] = torch.tensor(node_labels)  # N
        graph.ndata["masks"] = torch.tensor(node_masks)  # N
        graph.ndata["coordinates"] = torch.FloatTensor(node_coordinates_normal)

        return graph

    def __len__(self):
        return len(self.graphs)

    @staticmethod
    def collate(batch):
        return dgl.batch(batch)


def build_loaders(config, mode):
    return torch.utils.data.DataLoader(
        GNNDataset(config, mode),
        batch_size=config.model_config.batch_size,
        num_workers=config.model_config.num_workers,
        shuffle=mode == "train",
        collate_fn=GNNDataset.collate,
    )
