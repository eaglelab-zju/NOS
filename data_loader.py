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
from utils import (
    get_coordinators,
    get_node_weight,
    get_screen_configs,
    is_focusable,
    is_valid,
)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode="train"):
        self.config = config
        self.ids = (
            self.config.dataset_config.train_data_ids
            if mode == "train"
            else self.config.dataset_config.valid_data_ids
        )
        self.data_file = (
            self.config.dataset_config.train_data_file
            if mode == "train"
            else self.config.dataset_config.valid_data_file
        )
        self.mode = mode
        self.load_data()

    def load_data(self):
        raise NotImplementedError()

    @staticmethod
    def collate(batch):
        return default_collate(batch)


class MLPDataset(BaseDataset):
    def __init__(self, config, mode="train"):
        super().__init__(config, mode)

    def load_data(self):
        (
            self.node_images,
            self.node_texts,
            self.node_attributes,
            self.node_coordinators_normal,
            self.node_coordinators,
            self.node_labels,
            self.node_weights,
        ) = ([], [], [], [], [], [], [])

        for page_id in tqdm(self.ids):
            with open(
                f"{self.config.dataset_config.hierarchy_dir}/{page_id}.json",
                "r",
                encoding="utf-8",
            ) as f:
                nodes = json.load(f)["nodes"]

            phone_height, phone_width, extra_bottom, length_threshold = (
                get_screen_configs(self.config.dataset_config, page_id)
            )
            for node in nodes:
                if not is_valid(
                    node,
                    phone_height,
                    phone_width,
                    extra_bottom,
                    length_threshold,
                    inplace=True,
                ):
                    continue

                self.node_images.append(
                    (
                        f"{self.config.dataset_config.box_feat_dir}/{page_id}.pt",
                        node["id"],
                    )
                )
                self.node_weights.append(get_node_weight(self.config, page_id, node))
                self.node_texts.append(
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
                self.node_attributes.append(node["attributes"])
                self.node_coordinators_normal.append(node["coordinator_normal"])
                self.node_coordinators.append(
                    get_coordinators(
                        node,
                        phone_height,
                        phone_width,
                        extra_bottom,
                        length_threshold,
                    )
                )
                self.node_labels.append(int(is_focusable(node)))

    def __getitem__(self, idx):
        text_feat_file, node_id = self.node_texts[idx]
        text_feats = torch.load(text_feat_file, map_location="cpu", weights_only=True)[
            node_id
        ]
        image_feat_file, node_id = self.node_images[idx]
        image = torch.load(image_feat_file, map_location="cpu", weights_only=True)[
            node_id
        ]

        return {
            "images": image,
            "texts": text_feats,
            "labels": torch.tensor(self.node_labels[idx]),
            "coordinators": torch.FloatTensor(self.node_coordinators_normal[idx]),
            "attributes": torch.FloatTensor(self.node_attributes[idx]),
            "weights": torch.FloatTensor(self.node_weights[idx]),
        }

    def __len__(self):
        return len(self.node_images)


class GNNDataset(DGLDataset):
    def __init__(self, config, mode="train"):
        self.config = config
        self.ids = (
            self.config.dataset_config.train_data_ids
            if mode == "train"
            else self.config.dataset_config.valid_data_ids
        )
        self.data_file = (
            self.config.dataset_config.train_data_file
            if mode == "train"
            else self.config.dataset_config.valid_data_file
        )
        self.mode = mode

        super().__init__(name="A11yGraphDatasetOnline")

    def process(self):
        (
            self.page_images,
            self.page_texts,
            self.page_attributes,
            self.page_coordinators_normal,
            self.page_coordinators,
            self.page_masks,
            self.page_labels,
            self.page_weights,
            self.graphs,
        ) = ([], [], [], [], [], [], [], [], [])
        for page_id in tqdm(self.ids):
            with open(
                f"{self.config.dataset_config.hierarchy_dir}/{page_id}.json",
                "r",
                encoding="utf-8",
            ) as f:
                nodes = json.load(f)["nodes"]

            # if len(nodes) > self.config.dataset_config.graph_node_num_thresold:
            #     continue

            (
                node_images,
                node_texts,
                node_attributes,
                node_coordinators_normal,
                node_coordinators,
                node_masks,
                node_labels,
                node_weights,
            ) = ([], [], [], [], [], [], [], [])
            phone_height, phone_width, extra_bottom, length_threshold = (
                get_screen_configs(self.config.dataset_config, page_id)
            )
            for node in nodes:
                node_masks.append(
                    int(
                        is_valid(
                            node,
                            phone_height,
                            phone_width,
                            extra_bottom,
                            length_threshold,
                            inplace=True,
                        )
                    )
                )
                node_weights.append(get_node_weight(self.config, page_id, node))
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
                node_coordinators_normal.append(node["coordinator_normal"])
                node_coordinators.append(
                    get_coordinators(
                        node,
                        phone_height,
                        phone_width,
                        extra_bottom,
                        length_threshold,
                    )
                )
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
            self.page_coordinators_normal.append(node_coordinators_normal)
            self.page_coordinators.append(node_coordinators)
            self.page_masks.append(node_masks)
            self.page_labels.append(node_labels)
            self.page_weights.append(node_weights)
            self.graphs.append(f"{self.config.dataset_config.graph_dir}/{page_id}.npy")

    def __getitem__(self, idx):
        (
            node_images,
            node_texts,
            node_attributes,
            node_coordinators_normal,
            node_coordinators,
            node_masks,
            node_labels,
            node_weights,
            graph,
        ) = (
            self.page_images[idx],
            self.page_texts[idx],
            self.page_attributes[idx],
            self.page_coordinators_normal[idx],
            self.page_coordinators[idx],
            self.page_masks[idx],
            self.page_labels[idx],
            self.page_weights[idx],
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

        max_n = self.config.dataset_config.graph_node_num_thresold
        graph = np.load(graph)[:max_n, :max_n]
        graph = nx.from_numpy_array(graph, create_using=nx.DiGraph).to_directed()
        graph = dgl.from_networkx(graph)
        graph = dgl.add_self_loop(graph)

        graph.ndata["images"] = torch.stack(images[:max_n], dim=0)  # N * 1000
        graph.ndata["texts"] = torch.stack(texts[:max_n], dim=0)  # N * (768 * 4)
        graph.ndata["attributes"] = torch.FloatTensor(node_attributes[:max_n])  # N * 11
        graph.ndata["labels"] = torch.tensor(node_labels[:max_n])  # N
        graph.ndata["masks"] = torch.tensor(node_masks[:max_n])  # N
        graph.ndata["coordinators"] = torch.FloatTensor(
            node_coordinators_normal[:max_n]
        )

        return graph

    def __len__(self):
        return len(self.graphs)

    @staticmethod
    def collate(batch):
        return dgl.batch(batch)


class TransformerDataset(BaseDataset):
    def __init__(self, config, mode):
        super().__init__(config, mode)

    def load_data(self):
        (
            self.page_images,
            self.page_texts,
            self.page_attributes,
            self.page_coordinators_normal,
            self.page_coordinators,
            self.page_attn_masks,
            self.page_loss_masks,
            self.page_labels,
            self.page_weights,
        ) = ([], [], [], [], [], [], [], [], [])
        for page_id in tqdm(self.ids):
            with open(
                f"{self.config.dataset_config.hierarchy_dir}/{page_id}.json",
                "r",
                encoding="utf-8",
            ) as f:
                nodes = json.load(f)["nodes"]

            valid_nodes = []
            phone_height, phone_width, extra_bottom, length_threshold = (
                get_screen_configs(self.config.dataset_config, page_id)
            )
            for node in nodes:
                if is_valid(
                    node,
                    phone_height,
                    phone_width,
                    extra_bottom,
                    length_threshold,
                ):
                    valid_nodes.append(node)
            # 节点切分，窗口重叠
            for i in range(
                0,
                len(valid_nodes),
                self.config.model_config.seq_length
                - self.config.model_config.seq_overlap * 2,
            ):
                # [0, seq_overlap - 1][seq_overlap, seq_length - seq_overlap - 1][seq_length - seq_overlap, seq_length - 1]
                seq_nodes = valid_nodes[i : i + self.config.model_config.seq_length]
                if i > 0 and len(seq_nodes) <= self.config.model_config.seq_overlap:
                    continue

                (
                    node_images,
                    node_texts,
                    node_attributes,
                    node_coordinators_normal,
                    node_coordinators,
                    node_attn_masks,
                    node_loss_masks,
                    node_labels,
                    node_weights,
                ) = ([], [], [], [], [], [], [], [], [])
                for j, node in enumerate(seq_nodes):
                    node_images.append(
                        (
                            f"{self.config.dataset_config.box_feat_dir}/{page_id}.pt",
                            node["id"],
                        )
                    )
                    node_weights.append(get_node_weight(self.config, page_id, node))
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
                    node_coordinators_normal.append(node["coordinator_normal"])
                    node_coordinators.append(
                        get_coordinators(
                            node,
                            phone_height,
                            phone_width,
                            extra_bottom,
                            length_threshold,
                        )
                    )
                    node_labels.append(int(is_focusable(node)))
                    node_attn_masks.append(False)
                    # 前窗口
                    if i > 0 and j < self.config.model_config.seq_overlap:
                        node_loss_masks.append(0)
                    # 后窗口
                    elif (
                        j
                        >= self.config.model_config.seq_length
                        - self.config.model_config.seq_overlap
                    ):
                        node_loss_masks.append(0)
                    else:
                        node_loss_masks.append(1)

                # 填充数据
                while len(node_images) < self.config.model_config.seq_length:
                    node_images.append((None, None))
                    node_weights.append(0)
                    node_texts.append((None, None))
                    node_attributes.append([0] * 11)
                    node_coordinators_normal.append([0] * 4)
                    node_coordinators.append([0] * 4)
                    node_labels.append(0)
                    node_attn_masks.append(True)  # 这里得是 True 或非零
                    node_loss_masks.append(0)

                self.page_images.append(node_images)
                self.page_texts.append(node_texts)
                self.page_attributes.append(node_attributes)
                self.page_coordinators_normal.append(node_coordinators_normal)
                self.page_coordinators.append(node_coordinators)
                self.page_attn_masks.append(node_attn_masks)
                self.page_loss_masks.append(node_loss_masks)
                self.page_labels.append(node_labels)
                self.page_weights.append(node_weights)

    def __getitem__(self, idx):
        (
            node_images,
            node_texts,
            node_attributes,
            node_coordinators_normal,
            node_coordinators,
            node_attn_masks,
            node_loss_masks,
            node_labels,
            node_weights,
        ) = (
            self.page_images[idx],
            self.page_texts[idx],
            self.page_attributes[idx],
            self.page_coordinators_normal[idx],
            self.page_coordinators[idx],
            self.page_attn_masks[idx],
            self.page_loss_masks[idx],
            self.page_labels[idx],
            self.page_weights[idx],
        )

        page_text_feats = torch.load(
            node_texts[0][0], map_location="cpu", weights_only=True
        )
        page_image_feats = torch.load(
            node_images[0][0], map_location="cpu", weights_only=True
        )
        images, texts = [], []
        for image_file, node_id in node_images:
            if image_file is None:
                images.append(torch.zeros_like(images[-1]))
                texts.append(torch.zeros_like(texts[-1]))
            else:
                images.append(page_image_feats[node_id])
                texts.append(page_text_feats[node_id])

        return {
            "images": torch.stack(images, dim=0),
            "texts": torch.stack(texts, dim=0),
            "labels": torch.tensor(node_labels),
            "coordinators": torch.FloatTensor(node_coordinators_normal),
            "attributes": torch.FloatTensor(node_attributes),
            "attn_masks": torch.tensor(node_attn_masks),
            "loss_masks": torch.FloatTensor(node_loss_masks),
            "weights": torch.FloatTensor(node_weights),
        }

    def __len__(self):
        return len(self.page_images)


class DETRDataset(BaseDataset):
    def __init__(self, config, mode):
        super().__init__(config, mode)

        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.config.dataset_config.image_transform_resize),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load_data(self):
        (
            self.page_images,
            self.page_texts,
            self.page_attributes,
            self.page_coordinators_normal,
            self.page_coordinators,
            self.page_attn_masks,
            self.page_loss_masks,
            self.page_labels,
            self.page_weights,
        ) = ([], [], [], [], [], [], [], [], [])
        for page_id in tqdm(self.ids):
            with open(
                f"{self.config.dataset_config.hierarchy_dir}/{page_id}.json",
                "r",
                encoding="utf-8",
            ) as f:
                nodes = json.load(f)["nodes"]

            valid_nodes = []
            phone_height, phone_width, extra_bottom, length_threshold = (
                get_screen_configs(self.config.dataset_config, page_id)
            )
            for node in nodes:
                if is_valid(
                    node,
                    phone_height,
                    phone_width,
                    extra_bottom,
                    length_threshold,
                ):
                    valid_nodes.append(node)
            # 节点切分，窗口重叠
            for i in range(
                0,
                len(valid_nodes),
                self.config.model_config.seq_length
                - self.config.model_config.seq_overlap * 2,
            ):
                # [0, seq_overlap - 1][seq_overlap, seq_length - seq_overlap - 1][seq_length - seq_overlap, seq_length - 1]
                seq_nodes = valid_nodes[i : i + self.config.model_config.seq_length]
                if i > 0 and len(seq_nodes) <= self.config.model_config.seq_overlap:
                    continue

                (
                    node_texts,
                    node_attributes,
                    node_coordinators_normal,
                    node_coordinators,
                    node_attn_masks,
                    node_loss_masks,
                    node_labels,
                    node_weights,
                ) = ([], [], [], [], [], [], [], [])
                for j, node in enumerate(seq_nodes):
                    node_weights.append(get_node_weight(self.config, page_id, node))
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
                    node_coordinators_normal.append(node["coordinator_normal"])
                    node_coordinators.append(
                        get_coordinators(
                            node,
                            phone_height,
                            phone_width,
                            extra_bottom,
                            length_threshold,
                        )
                    )
                    node_labels.append(int(is_focusable(node)))
                    node_attn_masks.append(False)
                    # 前窗口
                    if i > 0 and j < self.config.model_config.seq_overlap:
                        node_loss_masks.append(0)
                    # 后窗口
                    elif (
                        j
                        >= self.config.model_config.seq_length
                        - self.config.model_config.seq_overlap
                    ):
                        node_loss_masks.append(0)
                    else:
                        node_loss_masks.append(1)

                # 填充数据
                while len(node_texts) < self.config.model_config.seq_length:
                    node_weights.append(0)
                    node_texts.append((None, None))
                    node_attributes.append([0] * 11)
                    node_coordinators_normal.append([0] * 4)
                    node_coordinators.append([0] * 4)
                    node_labels.append(0)
                    node_attn_masks.append(True)
                    node_loss_masks.append(0)

                self.page_images.append(
                    f"{self.config.dataset_config.image_dir}/{page_id}.png"
                )
                self.page_texts.append(node_texts)
                self.page_attributes.append(node_attributes)
                self.page_coordinators_normal.append(node_coordinators_normal)
                self.page_coordinators.append(node_coordinators)
                self.page_attn_masks.append(node_attn_masks)
                self.page_loss_masks.append(node_loss_masks)
                self.page_labels.append(node_labels)
                self.page_weights.append(node_weights)

    def __getitem__(self, idx):
        (
            node_image,
            node_texts,
            node_attributes,
            node_coordinators_normal,
            node_coordinators,
            node_attn_masks,
            node_loss_masks,
            node_labels,
            node_weights,
        ) = (
            self.page_images[idx],
            self.page_texts[idx],
            self.page_attributes[idx],
            self.page_coordinators_normal[idx],
            self.page_coordinators[idx],
            self.page_attn_masks[idx],
            self.page_loss_masks[idx],
            self.page_labels[idx],
            self.page_weights[idx],
        )
        image = cv2.imread(node_image)
        image = self.image_transform(image)

        texts = []
        page_text_feats = torch.load(
            node_texts[0][0], map_location="cpu", weights_only=True
        )
        for feat_path, node_id in node_texts:
            if feat_path is None:
                texts.append(
                    torch.zeros(
                        self.config.model_config.text_pretrained_model_output_dim
                    )
                )
            else:
                texts.append(page_text_feats[node_id])

        return {
            "images": image,
            "texts": torch.stack(texts, dim=0),
            "labels": torch.tensor(node_labels),
            "coordinators": torch.FloatTensor(node_coordinators_normal),
            "attributes": torch.FloatTensor(node_attributes),
            "attn_masks": torch.tensor(node_attn_masks),
            "loss_masks": torch.FloatTensor(node_loss_masks),
            "weights": torch.FloatTensor(node_weights),
        }

    def __len__(self):
        return len(self.page_images)


def build_loaders(config, mode, dataset_class):
    return torch.utils.data.DataLoader(
        dataset_class(config, mode),
        batch_size=config.model_config.batch_size,
        num_workers=config.model_config.num_workers,
        shuffle=mode == "train",
        collate_fn=dataset_class.collate,
    )
