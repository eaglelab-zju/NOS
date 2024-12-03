import json
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from utils import is_valid

device = torch.device("cuda:4")
# dataset_dir = "./dataset/nos-raw-labeled"
dataset_dir = "./dataset/rico-labeled"

if "nos-raw" in dataset_dir:
    phone_height, phone_width, extra_bottom, length_threshold = 1600, 720, 78, 5
elif "rico" in dataset_dir:
    phone_height, phone_width, extra_bottom, length_threshold = 2560, 1440, 168, 5


bert_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(bert_name)
text_pretrained_model = BertModel.from_pretrained(bert_name)
text_pretrained_model.eval()
text_pretrained_model.to(device)

image_pretrained_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).to(
    device
)
image_pretrained_model.eval()

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_coordinates(node):
    assert node["is_valid"]
    # 指定框的坐标和大小
    left, right, top, bottom = (
        node["screen_left"],
        node["screen_right"],
        node["screen_top"],
        node["screen_bottom"],
    )

    # 修正溢出框
    top = max(top, 0)
    bottom = min(bottom, phone_height - extra_bottom)
    left = max(left, 0)
    right = min(right, phone_width)

    (
        node["screen_left"],
        node["screen_right"],
        node["screen_top"],
        node["screen_bottom"],
    ) = (left, right, top, bottom)

    return [
        node["screen_left"],
        node["screen_right"],
        node["screen_top"],
        node["screen_bottom"],
    ]


def save_graph(nodes, page_id):
    node_num = len(nodes)
    graph = np.zeros((node_num, node_num))

    for node in nodes[1:]:
        graph[node["father"]][node["id"]] = 1

    np.save(f"{dataset_dir}/graph/{page_id}.npy", graph)


def save_text_embedding(nodes, page_id):
    texts = []
    if len(nodes) > 2000:
        text_pretrained_model.to(device)
    for node in nodes:
        texts += node["texts_info"]

    interval = 256
    text_embeddings = []
    for i in range(0, len(texts), interval):
        with torch.no_grad():
            encoded_inputs = tokenizer(
                texts[i : i + interval],
                add_special_tokens=True,
                padding="max_length",
                max_length=10,
                return_tensors="pt",
                truncation=True,
            ).to(device)

            # [N, H]
            text_embedding = text_pretrained_model(**encoded_inputs).last_hidden_state[
                :, 0, :
            ]
        text_embeddings.append(text_embedding.cpu())

    text_embeddings = torch.cat(
        text_embeddings, dim=0
    )  # N * 768 --> (N / 4) * (768 * 4)
    text_embeddings = text_embeddings.reshape(-1, 4, 768)
    text_embeddings, _ = torch.max(text_embeddings, dim=1)

    torch.save(text_embeddings, f"{dataset_dir}/text_feat/{page_id}.pt")


def save_box_embedding(image, nodes, page_id):
    image_height, image_width = image.shape[:2]
    images = []
    for node in nodes:
        img = image.copy()
        if not is_valid(node):
            box_image = torch.zeros((3, 224, 224))
        else:
            left, right, top, bottom = get_coordinates(node)
            left = int(left * image_width / phone_width)
            right = int(right * image_width / phone_width)
            top = int(top * image_height / phone_height)
            bottom = int(bottom * image_height / phone_height)
            box_image = img[top:bottom, left:right]
            cv2.imwrite(f"{dataset_dir}/box/{page_id}_{node['id']}.png", box_image)
            box_image = transform(box_image)
        images.append(box_image)
    images = torch.stack(images, dim=0).to(device)
    with torch.no_grad():
        total_images = []
        interval = 1024
        for i in range(0, len(images), 1024):
            cur_images = images[i : i + interval]
            cur_images = image_pretrained_model(cur_images)
            total_images.append(cur_images)

        total_images = torch.cat(total_images, dim=0)
    torch.save(total_images, f"{dataset_dir}/box_feat/{page_id}.pt")


def save_json(nodes, page_id):
    for node in nodes:
        # 归一化坐标 [left, top, right, bottom]
        node["coordinate_normal"] = [
            round(node["screen_left"] / phone_width, 2),
            round(node["screen_top"] / phone_height, 2),
            round(node["screen_right"] / phone_width, 2),
            round(node["screen_bottom"] / phone_height, 2),
        ]

        node["attributes"] = [
            1 if "focusable" in node and node["focusable"] else 0,
            1 if "checkable" in node and node["checkable"] else 0,
            1 if "checked" in node and node["checked"] else 0,
            1 if "focused" in node and node["focused"] else 0,
            1 if "selected" in node and node["selected"] else 0,
            1 if "clickable" in node and node["clickable"] else 0,
            1 if "long_clickable" in node and node["long_clickable"] else 0,
            1 if "context_clickable" in node and node["context_clickable"] else 0,
            1 if "enabled" in node and node["enabled"] else 0,
            1 if "text" in node and node["text"] else 0,
            1 if "content_description" in node and node["content_description"] else 0,
        ]

        node["texts_info"] = [
            node["text"] if "text" in node and node["text"] else "none",
            (
                node["class_name"].replace(".", " ").lower()
                if "class_name" in node and node["class_name"]
                else "none"
            ),
            (
                node["view_id_resource_name"]
                .replace(".", " ")
                .replace(":", " ")
                .replace("/", " ")
                .replace("_", " ")
                .lower()
                if "view_id_resource_name" in node and node["view_id_resource_name"]
                else "none"
            ),
            (node["content_description"] if node["content_description"] else "none"),
        ]

    with open(f"{dataset_dir}/hierarchy/{page_id}.json", "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes}, f)

    return nodes


def main():
    for subdir in ["graph", "box", "box_feat", "text_feat"]:
        if not os.path.exists(f"{dataset_dir}/{subdir}"):
            os.mkdir(f"{dataset_dir}/{subdir}")

    for file in tqdm(os.listdir(f"{dataset_dir}/hierarchy")):
        with open(f"{dataset_dir}/hierarchy/{file}", "r", encoding="utf-8") as f:
            nodes = json.load(f)["nodes"]
        page_id = int(file.split(".")[0])
        image = cv2.imread(f"{dataset_dir}/screenshot/{page_id}.png")

        save_json(nodes, page_id)
        save_graph(nodes, page_id)
        save_text_embedding(nodes, page_id)
        save_box_embedding(image, nodes, page_id)
        # break


if __name__ == "__main__":
    main()
