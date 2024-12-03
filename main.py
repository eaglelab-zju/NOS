import argparse
import multiprocessing
import os
import random

import numpy as np
import torch
import torchmetrics

from config import Config
from IGNN.ignn.models import FlatGNN


def seed_torch(seed=7777):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-M",
        "--mode",
        choices=["train", "predict"],
        default="train",
        help="select the mode",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        choices=["nos-raw", "rico", "mixed"],
        default="nos-raw",
        help="select the dataset",
    )
    parser.add_argument(
        "--split",
        choices=["1", "2", "3"],
        default="1",
        help="select the dataset split",
    )
    parser.add_argument(
        "-g", "--gpu", default="0", help="select the gpu device, default 0."
    )
    parser.add_argument(
        "-cp", "--checkpoint", default="", help="choose the checkpoint file id"
    )

    return parser.parse_args()


if __name__ == "__main__":
    seed_torch()
    args = parse_args()
    config = Config()
    multiprocessing.set_start_method("spawn")
    torch.backends.cudnn.enabled = False

    import shutil

    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm

    from config import (
        IGNNModelConfig,
        MixedDatasetConfig,
        NOSRawDatasetConfig,
        RicoDatasetConfig,
    )
    from data_loader import build_loaders
    from model import IGNN, MLP, GNNModel
    from predict import BaseDataLoader, GNNDataCollector, Predictor
    from train import train

    DATASET_CONFIG_MAP = {
        "nos-raw": NOSRawDatasetConfig,
        "rico": RicoDatasetConfig,
        "mixed": MixedDatasetConfig,
    }

    config.load_config(
        args,
        IGNNModelConfig(),
        DATASET_CONFIG_MAP[args.dataset](args.split),
    )
    config.model_config.device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    config.done()

    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    image_mlp_model = MLP(
        config,
        config.model_config.image_mlp_input_dim,
        config.model_config.image_mlp_hidden_dims,
        config.model_config.image_mlp_output_dim,
        config.model_config.image_mlp_dropout,
    )
    text_mlp_model = MLP(
        config,
        config.model_config.text_mlp_input_dim,
        config.model_config.text_mlp_hidden_dims,
        config.model_config.text_mlp_output_dim,
        config.model_config.text_mlp_dropout,
    )
    attribute_mlp_model = MLP(
        config,
        config.model_config.attribute_mlp_input_dim,
        config.model_config.attribute_mlp_hidden_dims,
        config.model_config.attribute_mlp_output_dim,
        config.model_config.attribute_mlp_dropout,
    )
    coordinate_mlp_model = MLP(
        config,
        config.model_config.coordinate_mlp_input_dim,
        config.model_config.coordinate_mlp_hidden_dims,
        config.model_config.coordinate_mlp_output_dim,
        config.model_config.coordinate_mlp_dropout,
    )
    combined_model = MLP(
        config,
        config.model_config.combined_mlp_input_dim,
        config.model_config.combined_mlp_hidden_dims,
        config.model_config.combined_mlp_output_dim,
        config.model_config.combined_mlp_dropout,
    )

    ignn_model = FlatGNN(
        in_feats=config.model_config.ignn_model_in_feats,
        h_feats=config.model_config.ignn_model_h_feats,
        n_clusters=config.model_config.classifier_model_output_dim,
        n_epochs=config.model_config.ignn_model_n_epochs,
        lr=config.model_config.ignn_model_lr,
        l2_coef=config.model_config.ignn_model_l2_coef,
        early_stop=config.model_config.ignn_model_early_stop,
        device=config.model_config.device,
        nas_dropout=config.model_config.ignn_model_nas_dropout,
        nss_dropout=config.model_config.ignn_model_nss_dropout,
        clf_dropout=config.model_config.ignn_model_clf_dropout,
        out_ndim_trans=config.model_config.ignn_model_out_ndim_trans,
        lda=config.model_config.ignn_model_lda,
        n_hops=config.model_config.ignn_model_n_hops,
        n_intervals=config.model_config.ignn_model_n_intervals,
        nie=config.model_config.ignn_model_nie,
        nrl=config.model_config.ignn_model_nrl,
        n_layers=config.model_config.ignn_model_ignn_layer_num,
        act=config.model_config.ignn_model_act,
        layer_norm=config.model_config.ignn_model_layer_norm,
        loss=config.model_config.ignn_model_loss,
        n_nodes=config.model_config.ignn_model_n_nodes,
        ndim_h_a=config.model_config.ignn_model_ndim_h_a,
        num_heads=config.model_config.ignn_model_num_heads,
        transform_first=config.model_config.ignn_model_transform_first,
        trans_layer_num=config.model_config.ignn_model_trans_layer_num,
        no_save=config.model_config.ignn_model_no_save,
    )
    gnn_model = IGNN(config, ignn_model)
    model = GNNModel(
        config,
        gnn_model,
        image_mlp_model,
        text_mlp_model,
        attribute_mlp_model,
        coordinate_mlp_model,
        combined_model,
    ).to(config.model_config.device)

    dataloaders = {
        "train": build_loaders(
            config,
            mode="train",
        ),
        "valid": build_loaders(
            config,
            mode="valid",
        ),
    }

    if args.mode == "train":
        # 设置损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(), lr=config.model_config.lr, momentum=0.9, nesterov=True
        )
        # 训练模型
        train(
            model,
            config,
            criterion,
            optimizer,
            config.model_config.num_epochs,
            dataloaders,
        )
    elif args.mode == "predict":
        model.load_state_dict(
            torch.load(config.checkpoint_file, map_location=config.model_config.device)
        )

        page_range = config.dataset_config.test_data_ids
        if os.path.exists(config.dataset_config.predict_result_dir):
            shutil.rmtree(config.dataset_config.predict_result_dir)
        os.makedirs(config.dataset_config.predict_result_dir)

        predictor = Predictor(config, model)
        with torch.no_grad():
            total = right = 0
            page_pred_map = {}
            test_acc = torchmetrics.Accuracy(task="binary")
            test_recall = torchmetrics.Recall(
                task="binary", average="none", num_classes=2
            )
            test_precision = torchmetrics.Precision(
                task="binary", average="none", num_classes=2
            )
            test_auc = torchmetrics.AUROC(task="binary", average="macro", num_classes=2)
            test_f1 = torchmetrics.F1Score(
                task="binary", average="macro", num_classes=2
            )
            for page_id in tqdm(page_range):
                loader = BaseDataLoader(config, page_id)
                collector = GNNDataCollector(
                    config,
                    page_id,
                    test_acc,
                    test_recall,
                    test_precision,
                    test_auc,
                    test_f1,
                )
                predictor.predict(loader, collector)
                total += collector.total
                right += collector.right
                print(f"total: {total}, right: {right}, acc: {right/total}")

            total_acc = test_acc.compute()
            total_recall = test_recall.compute()
            total_precision = test_precision.compute()
            total_auc = test_auc.compute()
            total_f1 = test_f1.compute()
            print(
                f"\nacc: {(100 * total_acc.item()):>0.2f}%, recall: {(100 * total_recall.item()):>0.2f}, precision: {(100 * total_precision.item()):>0.2f}, auc: {(100 * total_auc.item()):>0.2f}, f1: {(100 * total_f1.item()):>0.2f}"
            )
