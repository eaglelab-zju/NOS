import json
import os

import cv2
import dgl
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils import (
    get_coordinators,
    get_node_weight,
    is_focusable,
    is_valid,
    get_screen_configs,
)


class BaseDataLoader:
    def __init__(self, config, page_id):
        self.config = config
        self.page_id = page_id

    def load(self):
        raise NotImplementedError()


class BaseDataCollector:
    def __init__(
        self, config, page_id, test_acc, test_recall, test_precision, test_auc, test_f1
    ):
        self.config = config
        self.page_id = page_id

        self.total = 0
        self.right = 0

        self.test_acc = test_acc
        self.test_recall = test_recall
        self.test_precision = test_precision
        self.test_auc = test_auc
        self.test_f1 = test_f1

    def collect(self, image, nodes, model):
        raise NotImplementedError()


class Predictor:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.model.eval()

    def predict(self, loader: BaseDataLoader, collector: BaseDataCollector):
        image, nodes = loader.load()
        if not nodes:
            print(f"[Predict] {loader.page_id}: no nodes")
            return

        nodes = collector.collect(image, nodes, self.model)
        if not nodes:
            print(f"[Predict] {loader.page_id}: no valid nodes")
            return

        self.draw(
            loader.page_id,
            nodes,
            image,
            f"{self.config.dataset_config.predict_result_dir}/{loader.page_id}.png",
        )

    def draw(self, page_id, nodes, image, output_file):
        img1, img2, img3 = image.copy(), image.copy(), image.copy()
        thickness = 4
        height, width = img1.shape[:2]

        if self.config.args.dataset == "mixed":
            if page_id >= 0:
                phone_height, phone_width = (
                    self.config.dataset_config.a11y_phone_height,
                    self.config.dataset_config.a11y_phone_width,
                )
            else:
                phone_height, phone_width = (
                    self.config.dataset_config.rico_phone_height,
                    self.config.dataset_config.rico_phone_width,
                )
        else:
            phone_height, phone_width = (
                self.config.dataset_config.phone_height,
                self.config.dataset_config.phone_width,
            )

        y_ratio, x_ratio = (
            height / phone_height,
            width / phone_width,
        )

        for img, prop, text, color in zip(
            (img1, img2, img3),
            ("predict_focusable", "talkback_focusable", "actual_focusable"),
            ("Predict", "Talkback", "Ground Truth"),
            ((0, 255, 0), (0, 255, 0), (0, 255, 0)),
        ):
            for node in nodes:
                if not node.get(prop, False):
                    continue

                left, right, top, bottom = (
                    int(x_ratio * node["screen_left"]),
                    int(x_ratio * node["screen_right"]),
                    int(y_ratio * node["screen_top"]),
                    int(y_ratio * node["screen_bottom"]),
                )

                cur_color = color
                if self.config.args.dataset == "rico" or (
                    self.config.args.dataset == "mixed" and page_id < 0
                ):
                    # 假阳性
                    if prop == "predict_focusable" and not node.get(
                        "actual_focusable", False
                    ):
                        cur_color = (255, 0, 0)
                    # 假阴性
                    if prop == "actual_focusable" and not node.get(
                        "predict_focusable", False
                    ):
                        cur_color = (0, 0, 255)
                # 在指定坐标处画矩形框
                cv2.rectangle(img, (left, top), (right, bottom), cur_color, thickness)
            cv2.putText(
                img,
                text,
                (
                    int(phone_width / 2 - (100 if text == "Ground Truth" else 70)),
                    40,
                ),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        # 横向拼接
        if self.config.args.dataset == "rico" or (
            self.config.args.dataset == "mixed" and page_id < 0
        ):
            image = np.concatenate([img1, img3], axis=1)
        else:
            image = np.concatenate([img1, img2, img3], axis=1)
        cv2.imwrite(output_file, image)


class RicoDataLoader(BaseDataLoader):
    def load(self):
        page_screenshot_file = (
            f"{self.config.dataset_config.image_dir}/{self.page_id}.png"
        )
        if not os.path.exists(page_screenshot_file):
            return None, None

        image = cv2.imread(page_screenshot_file)
        with open(
            f"{self.config.dataset_config.hierarchy_dir}/{self.page_id}.json",
            "r",
            encoding="utf-8",
        ) as f:
            nodes = json.load(f)["nodes"]
        for node in nodes:
            node["manual_focusable"] = node.get("focusable_manual_label", False)
            # TODO: talkback
        return image, nodes


class A11yDataLoader(BaseDataLoader):
    def load(self):
        page_screenshot_file = (
            f"{self.config.dataset_config.image_dir}/{self.page_id}.png"
        )
        if not os.path.exists(page_screenshot_file):
            return None, None

        image = cv2.imread(page_screenshot_file)
        with open(
            f"{self.config.dataset_config.hierarchy_dir}/{self.page_id}.json",
            "r",
            encoding="utf-8",
        ) as f:
            nodes = json.load(f)["nodes"]
        for node in nodes:
            node["manual_focusable"] = node.get("focusable_manual_label", False)
            node["talkback_focusable"] = (
                node["is_web_node"] and (node["focusable_isAccessibilityFocusable"])
            ) or (not node["is_web_node"] and node["focusable_shouldFocusNode"])
        return image, nodes


class MixedDataLoader(BaseDataLoader):
    def load(self):
        page_screenshot_file = (
            f"{self.config.dataset_config.image_dir}/{self.page_id}.png"
        )
        if not os.path.exists(page_screenshot_file):
            return None, None

        image = cv2.imread(page_screenshot_file)
        with open(
            f"{self.config.dataset_config.hierarchy_dir}/{self.page_id}.json",
            "r",
            encoding="utf-8",
        ) as f:
            nodes = json.load(f)["nodes"]
        for node in nodes:
            node["manual_focusable"] = node.get("focusable_manual_label", False)
            if self.page_id >= 0:
                node["talkback_focusable"] = (
                    node["is_web_node"] and (node["focusable_isAccessibilityFocusable"])
                ) or (not node["is_web_node"] and node["focusable_shouldFocusNode"])
        return image, nodes


class MLPDataCollector(BaseDataCollector):
    def collect(self, image, nodes, model):
        (
            valid_nodes,
            node_images,
            node_texts,
            node_attributes,
            node_coordinators_normal,
            node_labels,
            node_weights,
        ) = ([], [], [], [], [], [], [])
        page_text_feats = torch.load(
            f"{self.config.dataset_config.text_feat_dir}/{self.page_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        page_image_feats = torch.load(
            f"{self.config.dataset_config.box_feat_dir}/{self.page_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        phone_height, phone_width, extra_bottom, length_threshold = get_screen_configs(
            self.config.dataset_config, self.page_id
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

            valid_nodes.append(node)
            node_images.append(page_image_feats[node["id"]])
            node_weights.append(get_node_weight(self.config, self.page_id, node))
            node_texts.append(page_text_feats[node["id"]])
            node_attributes.append(node["attributes"])
            node_coordinators_normal.append(node["coordinator_normal"])
            node_labels.append(int(is_focusable(node)))

        if not valid_nodes:
            return valid_nodes

        interval = 256
        self.node_pred_map = {}
        for i in range(0, len(valid_nodes), interval):
            batch = {
                "images": torch.stack(node_images[i : i + interval], dim=0).to(
                    self.config.model_config.device
                ),
                "texts": torch.stack(node_texts[i : i + interval], dim=0).to(
                    self.config.model_config.device
                ),
                "attributes": torch.FloatTensor(node_attributes[i : i + interval]).to(
                    self.config.model_config.device
                ),
                "coordinators": torch.FloatTensor(
                    node_coordinators_normal[i : i + interval]
                ).to(self.config.model_config.device),
                "labels": torch.FloatTensor(node_labels[i : i + interval]).to(
                    self.config.model_config.device
                ),
            }
            outputs = model(batch)
            _, preds = torch.max(outputs, dim=-1)

            for j, pred in enumerate(preds):
                valid_nodes[i + j]["predict_focusable"] = pred == 1
                valid_nodes[i + j]["actual_focusable"] = valid_nodes[i + j][
                    "manual_focusable"
                ]

            with torch.no_grad():
                outputs = torch.softmax(outputs, dim=-1)
                outputs = outputs[:, 1].to("cpu")
            preds = preds.to("cpu")
            labels = batch["labels"].to("cpu")
            sz = batch["images"].shape[0]
            self.total += sz
            self.right = torch.sum(preds == labels)

            self.test_acc.update(preds, labels)
            self.test_auc.update(outputs, labels)
            self.test_recall.update(preds, labels)
            self.test_precision.update(preds, labels)
            self.test_f1.update(preds, labels)
            total_acc = self.test_acc.compute()
            total_recall = self.test_recall.compute()
            total_precision = self.test_precision.compute()
            total_auc = self.test_auc.compute()
            total_f1 = self.test_f1.compute()
        print(
            f"\nacc: {(100 * total_acc.item()):>0.2f}%, recall: {(100 * total_recall.item()):>0.2f}, precision: {(100 * total_precision.item()):>0.2f}, auc: {(100 * total_auc.item()):>0.2f}, f1: {(100 * total_f1.item()):>0.2f}"
        )

        return valid_nodes


class GNNDataCollector(BaseDataCollector):
    def __init__(
        self, config, page_id, test_acc, test_recall, test_precision, test_auc, test_f1
    ):
        super().__init__(
            config, page_id, test_acc, test_recall, test_precision, test_auc, test_f1
        )

    def collect(self, image, nodes, model):
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
        page_text_feats = torch.load(
            f"{self.config.dataset_config.text_feat_dir}/{self.page_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        page_image_feats = torch.load(
            f"{self.config.dataset_config.box_feat_dir}/{self.page_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        phone_height, phone_width, extra_bottom, length_threshold = get_screen_configs(
            self.config.dataset_config, self.page_id
        )
        for node in nodes:
            node_weights.append(get_node_weight(self.config, self.page_id, node))
            node_texts.append(page_text_feats[node["id"]])
            node_attributes.append(node["attributes"])
            node_coordinators_normal.append(node["coordinator_normal"])
            node_coordinators.append(
                get_coordinators(
                    node, phone_height, phone_width, extra_bottom, length_threshold
                )
            )
            node_labels.append(int(is_focusable(node)))
            node_masks.append(
                is_valid(
                    node, phone_height, phone_width, extra_bottom, length_threshold
                )
            )
            node_images.append(page_image_feats[node["id"]])

        graph = np.load(f"{self.config.dataset_config.graph_dir}/{self.page_id}.npy")
        max_n = self.config.dataset_config.graph_node_num_thresold
        graph = graph[:max_n, :max_n]
        graph = nx.from_numpy_array(graph, create_using=nx.DiGraph).to_directed()
        graph = dgl.from_networkx(graph)
        graph = dgl.add_self_loop(graph)
        graph = graph.to(self.config.model_config.device)

        graph.ndata["images"] = torch.stack(node_images[:max_n], dim=0).to(
            self.config.model_config.device
        )  # N * 1000
        graph.ndata["texts"] = torch.stack(node_texts[:max_n], dim=0).to(
            self.config.model_config.device
        )  # N * (768 * 4)
        graph.ndata["attributes"] = torch.FloatTensor(node_attributes[:max_n]).to(
            self.config.model_config.device
        )  # N * 11
        graph.ndata["labels"] = torch.tensor(node_labels[:max_n]).to(
            self.config.model_config.device
        )  # N
        graph.ndata["masks"] = torch.tensor(node_masks[:max_n]).to(
            self.config.model_config.device
        )  # N
        graph.ndata["coordinators"] = torch.FloatTensor(
            node_coordinators_normal[:max_n]
        ).to(self.config.model_config.device)

        batch = dgl.batch([graph])
        outputs = model(batch)
        _, preds = torch.max(outputs, 1)
        for i, pred in enumerate(preds):
            if node_masks[i] == 0:
                continue
            nodes[i]["predict_focusable"] = pred == 1
            nodes[i]["actual_focusable"] = nodes[i]["manual_focusable"]

        with torch.no_grad():
            outputs = torch.softmax(outputs, dim=-1)
            outputs = outputs[:, 1][graph.ndata["masks"] > 0].to("cpu")
        preds = preds[graph.ndata["masks"] > 0].to("cpu")
        labels = graph.ndata["labels"][graph.ndata["masks"] > 0].to("cpu")
        self.test_acc.update(preds, labels)
        self.test_auc.update(outputs, labels)
        self.test_recall.update(preds, labels)
        self.test_precision.update(preds, labels)
        self.test_f1.update(preds, labels)
        total_acc = self.test_acc.compute()
        total_recall = self.test_recall.compute()
        total_precision = self.test_precision.compute()
        total_auc = self.test_auc.compute()
        total_f1 = self.test_f1.compute()
        self.total += torch.sum(graph.ndata["masks"] > 0)
        self.right += torch.sum(preds == labels)
        print(
            f"\nacc: {(100 * total_acc.item()):>0.2f}%, recall: {(100 * total_recall.item()):>0.2f}, precision: {(100 * total_precision.item()):>0.2f}, auc: {(100 * total_auc.item()):>0.2f}, f1: {(100 * total_f1.item()):>0.2f}"
        )

        return nodes


class TransformerDataCollector(BaseDataCollector):
    def collect(self, image, nodes, model):
        page_text_feats = torch.load(
            f"{self.config.dataset_config.text_feat_dir}/{self.page_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        page_image_feats = torch.load(
            f"{self.config.dataset_config.box_feat_dir}/{self.page_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        valid_nodes = []
        phone_height, phone_width, extra_bottom, length_threshold = get_screen_configs(
            self.config.dataset_config, self.page_id
        )
        for node in nodes:
            if is_valid(
                node, phone_height, phone_width, extra_bottom, length_threshold
            ):
                valid_nodes.append(node)

        for i in range(
            0,
            len(valid_nodes),
            self.config.model_config.seq_length
            - self.config.model_config.seq_overlap * 2,
        ):
            seq_nodes = valid_nodes[i : i + self.config.model_config.seq_length]
            if i > 0 and len(seq_nodes) <= self.config.model_config.seq_overlap:
                continue

            (
                node_images,
                node_texts,
                node_attributes,
                node_coordinators_normal,
                node_attn_masks,
                node_loss_masks,
                node_labels,
                node_weights,
            ) = ([], [], [], [], [], [], [], [])
            for j, node in enumerate(seq_nodes):
                node_images.append(page_image_feats[node["id"]])
                node_weights.append(get_node_weight(self.config, self.page_id, node))
                node_texts.append(page_text_feats[node["id"]])
                node_attributes.append(node["attributes"])
                node_coordinators_normal.append(node["coordinator_normal"])
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
            while len(node_images) < self.config.model_config.seq_length:
                node_weights.append(0)
                node_texts.append(torch.zeros(768))
                node_attributes.append([0] * 11)
                node_coordinators_normal.append([0] * 4)
                node_labels.append(0)
                node_attn_masks.append(True)
                node_loss_masks.append(0)
                node_images.append(torch.zeros_like(node_images[-1]))

            batch = {
                "images": torch.stack(node_images, dim=0)
                .unsqueeze(0)
                .to(self.config.model_config.device),
                "texts": torch.stack(node_texts, dim=0)
                .unsqueeze(0)
                .to(self.config.model_config.device),
                "coordinators": torch.FloatTensor(node_coordinators_normal)
                .unsqueeze(0)
                .to(self.config.model_config.device),
                "attributes": torch.FloatTensor(node_attributes)
                .unsqueeze(0)
                .to(self.config.model_config.device),
                "attn_masks": torch.tensor(node_attn_masks)
                .unsqueeze(0)
                .to(self.config.model_config.device),
                "loss_masks": torch.FloatTensor(node_loss_masks)
                .unsqueeze(0)
                .to(self.config.model_config.device),
                "labels": torch.tensor(node_labels).to(self.config.model_config.device),
            }
            loss_masks = batch["loss_masks"].squeeze(0)
            outputs = model(batch)
            _, preds = torch.max(outputs, dim=-1)
            # preds = (
            #     outputs[:, 1]
            #     >= self.config.dataset_config.classifier_positive_threshold
            # ).long()
            for j, pred in enumerate(preds):
                if loss_masks[j] == 0:
                    continue
                seq_nodes[j]["predict_focusable"] = pred == 1
                seq_nodes[j]["actual_focusable"] = seq_nodes[j]["manual_focusable"]

            with torch.no_grad():
                outputs = torch.softmax(outputs, dim=-1)
                outputs = outputs[:, 1][loss_masks > 0].to("cpu")
            preds = preds[loss_masks > 0].to("cpu")
            labels = batch["labels"][loss_masks > 0].to("cpu")
            self.test_acc.update(preds, labels)
            self.test_auc.update(outputs, labels)
            self.test_recall.update(preds, labels)
            self.test_precision.update(preds, labels)
            self.test_f1.update(preds, labels)
            self.total += torch.sum(loss_masks > 0)
            self.right += torch.sum(preds == labels)

        total_acc = self.test_acc.compute()
        total_recall = self.test_recall.compute()
        total_precision = self.test_precision.compute()
        total_auc = self.test_auc.compute()
        total_f1 = self.test_f1.compute()
        print(
            f"\nacc: {(100 * total_acc.item()):>0.2f}%, recall: {(100 * total_recall.item()):>0.2f}, precision: {(100 * total_precision.item()):>0.2f}, auc: {(100 * total_auc.item()):>0.2f}, f1: {(100 * total_f1.item()):>0.2f}"
        )

        return valid_nodes


class DETRDataCollector(BaseDataCollector):
    def collect(self, image, nodes, model):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.config.dataset_config.image_transform_resize),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        page_image = transform(image)
        page_text_feats = torch.load(
            f"{self.config.dataset_config.text_feat_dir}/{self.page_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        valid_nodes = []
        phone_height, phone_width, extra_bottom, length_threshold = get_screen_configs(
            self.config.dataset_config, self.page_id
        )
        for node in nodes:
            if is_valid(
                node, phone_height, phone_width, extra_bottom, length_threshold
            ):
                valid_nodes.append(node)

        for i in range(
            0,
            len(valid_nodes),
            self.config.model_config.seq_length
            - self.config.model_config.seq_overlap * 2,
        ):
            seq_nodes = valid_nodes[i : i + self.config.model_config.seq_length]
            if i > 0 and len(seq_nodes) <= self.config.model_config.seq_overlap:
                continue

            (
                node_texts,
                node_attributes,
                node_coordinators_normal,
                node_attn_masks,
                node_loss_masks,
                node_labels,
                node_weights,
            ) = ([], [], [], [], [], [], [])
            for j, node in enumerate(seq_nodes):
                node_weights.append(get_node_weight(self.config, self.page_id, node))
                node_texts.append(page_text_feats[node["id"]])
                node_attributes.append(node["attributes"])
                node_coordinators_normal.append(node["coordinator_normal"])
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
            while len(node_texts) < self.config.model_config.seq_length:
                node_weights.append(0)
                node_texts.append(torch.zeros(768))
                node_attributes.append([0] * 11)
                node_coordinators_normal.append([0] * 4)
                node_labels.append(0)
                node_attn_masks.append(True)
                node_loss_masks.append(0)

            batch = {
                "images": page_image.unsqueeze(0).to(self.config.model_config.device),
                "texts": torch.stack(node_texts, dim=0)
                .unsqueeze(0)
                .to(self.config.model_config.device),
                "coordinators": torch.FloatTensor(node_coordinators_normal)
                .unsqueeze(0)
                .to(self.config.model_config.device),
                "attributes": torch.FloatTensor(node_attributes)
                .unsqueeze(0)
                .to(self.config.model_config.device),
                "attn_masks": torch.tensor(node_attn_masks)
                .unsqueeze(0)
                .to(self.config.model_config.device),
                "loss_masks": torch.FloatTensor(node_loss_masks)
                .unsqueeze(0)
                .to(self.config.model_config.device),
                "labels": torch.FloatTensor(node_labels).to(
                    self.config.model_config.device
                ),
            }
            loss_masks = batch["loss_masks"].squeeze(0)
            outputs = model(batch)
            outputs = torch.flatten(outputs, start_dim=0, end_dim=1)
            _, preds = torch.max(outputs, dim=-1)
            for j, pred in enumerate(preds):
                if loss_masks[j] == 0:
                    continue
                seq_nodes[j]["predict_focusable"] = pred == 1
                seq_nodes[j]["actual_focusable"] = seq_nodes[j]["manual_focusable"]

            with torch.no_grad():
                outputs = torch.softmax(outputs, dim=-1)
                outputs = outputs[:, 1][loss_masks > 0].to("cpu")
            preds = preds[loss_masks > 0].to("cpu")
            labels = batch["labels"][loss_masks > 0].to("cpu")
            self.test_acc.update(preds, labels)
            self.test_auc.update(outputs, labels)
            self.test_recall.update(preds, labels)
            self.test_precision.update(preds, labels)
            self.test_f1.update(preds, labels)
            self.total += torch.sum(loss_masks > 0)
            self.right += torch.sum(preds == labels)

        total_acc = self.test_acc.compute()
        total_recall = self.test_recall.compute()
        total_precision = self.test_precision.compute()
        total_auc = self.test_auc.compute()
        total_f1 = self.test_f1.compute()
        print(
            f"\nacc: {(100 * total_acc.item()):>0.2f}%, recall: {(100 * total_recall.item()):>0.2f}, precision: {(100 * total_precision.item()):>0.2f}, auc: {(100 * total_auc.item()):>0.2f}, f1: {(100 * total_f1.item()):>0.2f}"
        )

        return valid_nodes
