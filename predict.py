import json
import os

import cv2
import dgl
import networkx as nx
import numpy as np
import torch

from utils import is_focusable, is_valid


class BaseDataLoader:
    def __init__(self, config, page_id):
        self.config = config
        self.page_id = page_id

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
        return image, nodes


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
            ("predict_focusable", "talkback_focusable", "label_focusable"),
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

                # 在指定坐标处画矩形框
                cv2.rectangle(img, (left, top), (right, bottom), color, thickness)
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
            node_coordinates_normal,
            node_masks,
            node_labels,
        ) = ([], [], [], [], [], [])
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
        for node in nodes:
            node_texts.append(page_text_feats[node["id"]])
            node_attributes.append(node["attributes"])
            node_coordinates_normal.append(node["coordinate_normal"])
            node_labels.append(int(is_focusable(node)))
            node_masks.append(int(is_valid(node)))
            node_images.append(page_image_feats[node["id"]])

        graph = np.load(f"{self.config.dataset_config.graph_dir}/{self.page_id}.npy")
        graph = nx.from_numpy_array(graph, create_using=nx.DiGraph).to_directed()
        graph = dgl.from_networkx(graph)
        graph = dgl.add_self_loop(graph)
        graph = graph.to(self.config.model_config.device)

        graph.ndata["images"] = torch.stack(node_images, dim=0).to(
            self.config.model_config.device
        )  # N * 1000
        graph.ndata["texts"] = torch.stack(node_texts, dim=0).to(
            self.config.model_config.device
        )  # N * (768 * 4)
        graph.ndata["attributes"] = torch.FloatTensor(node_attributes).to(
            self.config.model_config.device
        )  # N * 11
        graph.ndata["labels"] = torch.tensor(node_labels).to(
            self.config.model_config.device
        )  # N
        graph.ndata["masks"] = torch.tensor(node_masks).to(
            self.config.model_config.device
        )  # N
        graph.ndata["coordinates"] = torch.FloatTensor(node_coordinates_normal).to(
            self.config.model_config.device
        )

        batch = dgl.batch([graph])
        outputs = model(batch)
        _, preds = torch.max(outputs, 1)
        for i, pred in enumerate(preds):
            if node_masks[i] == 0:
                continue
            nodes[i]["predict_focusable"] = pred == 1

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
