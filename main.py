import argparse
import multiprocessing
import os
import random

import numpy as np
import torch
import torchmetrics

from config import Config
from IGNN.ignn.models import FlatGNN
from UGNN.semi_heter.baselines import GAT as UGNN_GAT
from UGNN.semi_heter.baselines import GCN as UGNN_GCN
from UGNN.semi_heter.baselines import SIGN as UGNN_SIGN
from UGNN.semi_heter.baselines import GraphSAGE as UGNN_GraphSAGE
from UGNN.semi_heter.baselines import OrderedGNN as UGNN_OrderedGNN
from UGNN.semi_heter.baselines import SGFormer as UGNN_SGFormer


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
        "-m",
        "--model",
        choices=["mlp", "gnn", "transformer"],
        default="mlp",
        help="select the model",
    )
    parser.add_argument(
        "-s",
        "--submodel",
        choices=[
            "gcn",
            "sage",
            "mlp",
            "transformer",
            "detr",
            "ignn",
            "gat",
            "sign",
            "ordered_gnn",
            "sg_former",
        ],
        default="mlp",
        help="select the sub model",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        choices=["a11y", "rico", "mixed"],
        default="a11y",
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
        A11yDatasetConfig,
        DETRModelConfig,
        GATModelConfig,
        GCNModelConfig,
        GraphSAGEModelConfig,
        IGNNModelConfig,
        MLPModelConfig,
        OrderedGNNModelConfig,
        RicoDatasetConfig,
        SGFormerModelConfig,
        SIGNModelConfig,
        TransformerModelConfig,
        MixedDatasetConfig
    )
    from data_loader import (
        DETRDataset,
        GNNDataset,
        MLPDataset,
        TransformerDataset,
        build_loaders,
    )
    from detr import build
    from model import (
        GAT,
        GCN,
        IGNN,
        MLP,
        SIGN,
        CrossAttentionEncoderModel,
        DETRModel,
        GNNModel,
        GraphSAGE,
        MLPModel,
        OrderedGNN,
        SGFormer,
        TransformerModel,
    )
    from predict import (
        A11yDataLoader,
        DETRDataCollector,
        GNNDataCollector,
        MLPDataCollector,
        Predictor,
        RicoDataLoader,
        TransformerDataCollector,
        MixedDataLoader
    )
    from train import train_detr, train_gnn, train_mlp, train_transformer

    DATA_LOADER_MAP = {
        "mlp": MLPDataset,
        "transformer": TransformerDataset,
        "detr": DETRDataset,
        "gcn": GNNDataset,
        "gat": GNNDataset,
        "sage": GNNDataset,
        "ignn": GNNDataset,
        "sign": GNNDataset,
        "ordered_gnn": GNNDataset,
        "sg_former": GNNDataset,
    }
    MODEL_MAP = {
        "mlp": MLPModel,
        "transformer": TransformerModel,
        "detr": DETRModel,
        "gcn": GNNModel,
        "gat": GNNModel,
        "sage": GNNModel,
        "ignn": GNNModel,
        "sign": GNNModel,
        "ordered_gnn": GNNModel,
        "sg_former": GNNModel,
    }
    MODEL_CONFIG_MAP = {
        "mlp": MLPModelConfig,
        "transformer": TransformerModelConfig,
        "detr": DETRModelConfig,
        "gcn": GCNModelConfig,
        "gat": GATModelConfig,
        "sage": GraphSAGEModelConfig,
        "ignn": IGNNModelConfig,
        "sign": SIGNModelConfig,
        "ordered_gnn": OrderedGNNModelConfig,
        "sg_former": SGFormerModelConfig,
    }
    DATASET_CONFIG_MAP = {"a11y": A11yDatasetConfig, "rico": RicoDatasetConfig, "mixed": MixedDatasetConfig}
    TRAIN_FUNC_MAP = {
        "mlp": train_mlp,
        "transformer": train_transformer,
        "detr": train_detr,
        "gcn": train_gnn,
        "gat": train_gnn,
        "sage": train_gnn,
        "ignn": train_gnn,
        "sign": train_gnn,
        "ordered_gnn": train_gnn,
        "sg_former": train_gnn,
    }
    CRITERION_MAP = {
        "mlp": nn.CrossEntropyLoss(),
        "transformer": nn.CrossEntropyLoss(),
        "detr": nn.CrossEntropyLoss(),
        "gcn": nn.CrossEntropyLoss(),
        "gat": nn.CrossEntropyLoss(),
        "sage": nn.CrossEntropyLoss(),
        "ignn": nn.CrossEntropyLoss(),
        "sign": nn.CrossEntropyLoss(),
        "ordered_gnn": nn.CrossEntropyLoss(),
        "sg_former": nn.CrossEntropyLoss(),
    }
    PREDICTOR_DATALOADER_MAP = {"a11y": A11yDataLoader, "rico": RicoDataLoader, "mixed": MixedDataLoader}
    PREDICTOR_DATACOLLECTOR_MAP = {
        "mlp": MLPDataCollector,
        "transformer": TransformerDataCollector,
        "detr": DETRDataCollector,
        "gcn": GNNDataCollector,
        "gat": GNNDataCollector,
        "sage": GNNDataCollector,
        "ignn": GNNDataCollector,
        "sign": GNNDataCollector,
        "ordered_gnn": GNNDataCollector,
        "sg_former": GNNDataCollector,
    }

    config.load_config(
        args,
        MODEL_CONFIG_MAP[args.submodel](),
        DATASET_CONFIG_MAP[args.dataset](args.split),
    )
    config.model_config.device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    config.done()

    if args.submodel == "mlp":
        image_mlp_model = MLP(
            config,
            config.model_config.image_mlp_input_dim,
            config.model_config.image_mlp_hidden_dims,
            config.model_config.image_mlp_output_dim,
            config.model_config.image_mlp_dropout,
            layer_norm=False,
        )
        text_mlp_model = MLP(
            config,
            config.model_config.text_mlp_input_dim,
            config.model_config.text_mlp_hidden_dims,
            config.model_config.text_mlp_output_dim,
            config.model_config.text_mlp_dropout,
            layer_norm=False,
        )
        attribute_mlp_model = MLP(
            config,
            config.model_config.attribute_mlp_input_dim,
            config.model_config.attribute_mlp_hidden_dims,
            config.model_config.attribute_mlp_output_dim,
            config.model_config.attribute_mlp_dropout,
            layer_norm=False,
        )
        coordinator_mlp_model = MLP(
            config,
            config.model_config.coordinator_mlp_input_dim,
            config.model_config.coordinator_mlp_hidden_dims,
            config.model_config.coordinator_mlp_output_dim,
            config.model_config.coordinator_mlp_dropout,
            layer_norm=False,
        )
        classifier_model = MLP(
            config,
            config.model_config.classifier_mlp_input_dim,
            config.model_config.classifier_mlp_hidden_dims,
            config.model_config.classifier_mlp_output_dim,
            config.model_config.classifier_mlp_dropout,
            layer_norm=False,
        )
        model = MODEL_MAP[args.submodel](
            config,
            image_mlp_model,
            text_mlp_model,
            attribute_mlp_model,
            coordinator_mlp_model,
            classifier_model,
        ).to(config.model_config.device)
    elif args.submodel == "transformer":
        text_mlp_model = MLP(
            config,
            config.model_config.text_mlp_model_input_dim,
            config.model_config.text_mlp_model_hidden_dims,
            config.model_config.text_mlp_model_output_dim,
            config.model_config.text_mlp_model_dropout,
            layer_norm=False,
        )
        attributes_model = MLP(
            config,
            config.model_config.attributes_model_input_dim,
            config.model_config.attributes_model_hidden_dims,
            config.model_config.attributes_model_output_dim,
            config.model_config.attributes_model_dropout,
            layer_norm=False,
        )
        coordinator_model = MLP(
            config,
            config.model_config.coordinator_model_input_dim,
            config.model_config.coordinator_model_hidden_dims,
            config.model_config.coordinator_model_output_dim,
            config.model_config.coordinator_model_dropout,
            layer_norm=False,
        )
        transformer_decoder_model = CrossAttentionEncoderModel(
            config,
            config.model_config.transformer_decoder_model_layer_input_dim,
            config.model_config.transformer_decoder_model_num_heads,
            config.model_config.transformer_decoder_model_num_layers,
            config.model_config.transformer_decoder_model_dim_feedforward,
            config.model_config.transformer_decoder_model_dropout,
        )
        classifier_model = MLP(
            config,
            config.model_config.classifier_model_input_dim,
            config.model_config.classifier_model_hidden_dims,
            config.model_config.classifier_model_output_dim,
            config.model_config.classifier_model_dropout,
            layer_norm=False,
        )
        model = MODEL_MAP[args.submodel](
            config,
            text_mlp_model,
            attributes_model,
            coordinator_model,
            transformer_decoder_model,
            classifier_model,
        ).to(config.model_config.device)
    elif args.submodel == "detr":
        text_mlp_model = MLP(
            config,
            config.model_config.text_mlp_model_input_dim,
            config.model_config.text_mlp_model_hidden_dims,
            config.model_config.text_mlp_model_output_dim,
            config.model_config.text_mlp_model_dropout,
        )
        attributes_model = MLP(
            config,
            config.model_config.attributes_model_input_dim,
            config.model_config.attributes_model_hidden_dims,
            config.model_config.attributes_model_output_dim,
            config.model_config.attributes_model_dropout,
        )
        coordinator_model = MLP(
            config,
            config.model_config.coordinator_model_input_dim,
            config.model_config.coordinator_model_hidden_dims,
            config.model_config.coordinator_model_output_dim,
            config.model_config.coordinator_model_dropout,
        )
        node_mlp_model = MLP(
            config,
            config.model_config.node_mlp_model_input_dim,
            config.model_config.node_mlp_model_hidden_dims,
            config.model_config.node_mlp_model_output_dim,
            config.model_config.node_mlp_model_dropout,
        )
        classifier_model = MLP(
            config,
            config.model_config.classifier_model_input_dim,
            config.model_config.classifier_model_hidden_dims,
            config.model_config.classifier_model_output_dim,
            config.model_config.classifier_model_dropout,
        )
        config.model_config.detr_args["device"] = config.model_config.device
        detr_model = build(argparse.Namespace(**config.model_config.detr_args))
        model = MODEL_MAP[args.submodel](
            config,
            text_mlp_model,
            attributes_model,
            coordinator_model,
            node_mlp_model,
            detr_model,
            classifier_model,
        ).to(config.model_config.device)
    elif args.model == "gnn":
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
        coordinator_mlp_model = MLP(
            config,
            config.model_config.coordinator_mlp_input_dim,
            config.model_config.coordinator_mlp_hidden_dims,
            config.model_config.coordinator_mlp_output_dim,
            config.model_config.coordinator_mlp_dropout,
        )
        combined_model = MLP(
            config,
            config.model_config.combined_mlp_input_dim,
            config.model_config.combined_mlp_hidden_dims,
            config.model_config.combined_mlp_output_dim,
            config.model_config.combined_mlp_dropout,
        )
        if args.submodel == "sage":
            graph_sage_model = UGNN_GraphSAGE(
                in_features=config.model_config.graph_sage_model_in_features,
                class_num=config.model_config.graph_sage_model_class_num,
                device=config.model_config.device,
                args=argparse.Namespace(**config.model_config.graph_sage_model_args),
            )
            gnn_model = GraphSAGE(config, graph_sage_model)
        elif args.submodel == "gcn":
            gcn_model = UGNN_GCN(
                in_features=config.model_config.gcn_model_in_features,
                class_num=config.model_config.gcn_model_class_num,
                device=config.model_config.device,
                args=argparse.Namespace(**config.model_config.gcn_model_args),
            )
            gnn_model = GCN(config, gcn_model)
        elif args.submodel == "gat":
            gat_model = UGNN_GAT(
                in_features=config.model_config.gat_model_in_features,
                class_num=config.model_config.gat_model_class_num,
                device=config.model_config.device,
                args=argparse.Namespace(**config.model_config.gat_model_args),
            )
            gnn_model = GAT(config, gat_model)
        elif args.submodel == "ignn":
            ignn_model = FlatGNN(
                in_feats=config.model_config.ignn_model_in_feats,
                h_feats=config.model_config.ignn_model_h_feats,
                n_clusters=config.model_config.classifier_model_output_dim,
                n_epochs=None,
                lr=1e-3,
                l2_coef=0.2,
                early_stop=None,
                device=config.model_config.device,
                nas_dropout=config.model_config.ignn_model_nas_dropout,
                nss_dropout=config.model_config.ignn_model_nss_dropout,
                clf_dropout=config.model_config.ignn_model_clf_dropout,
                out_ndim_trans=config.model_config.ignn_model_out_ndim_trans,
                lda=None,
                n_hops=config.model_config.ignn_model_n_hops,
                n_intervals=config.model_config.ignn_model_n_intervals,
                nie=config.model_config.ignn_model_nie,
                nrl=config.model_config.ignn_model_nrl,
                n_layers=config.model_config.ignn_model_ignn_layer_num,
                act=config.model_config.ignn_model_act,
                layer_norm=config.model_config.ignn_model_layer_norm,
                loss="ce",
                n_nodes=config.model_config.ignn_model_n_nodes,
                ndim_h_a=config.model_config.ignn_model_ndim_h_a,
                num_heads=config.model_config.ignn_model_num_heads,
                transform_first=config.model_config.ignn_model_transform_first,
                trans_layer_num=config.model_config.ignn_model_trans_layer_num,
                no_save=config.model_config.ignn_model_no_save,
            )
            gnn_model = IGNN(config, ignn_model)
        elif args.submodel == "sign":
            sign_model = UGNN_SIGN(
                in_features=config.model_config.sign_model_in_features,
                class_num=config.model_config.sign_model_class_num,
                device=config.model_config.device,
                args=argparse.Namespace(**config.model_config.sign_model_args),
            )
            gnn_model = SIGN(config, sign_model)
        elif args.submodel == "ordered_gnn":
            ordered_gnn_model = UGNN_OrderedGNN(
                in_features=config.model_config.ordered_gnn_model_in_features,
                class_num=config.model_config.ordered_gnn_model_class_num,
                device=config.model_config.device,
                args=argparse.Namespace(**config.model_config.ordered_gnn_model_args),
            )
            gnn_model = OrderedGNN(config, ordered_gnn_model)
        elif args.submodel == "sg_former":
            glo_gnn_model = UGNN_SGFormer(
                in_features=config.model_config.sg_former_model_in_features,
                class_num=config.model_config.sg_former_model_class_num,
                device=config.model_config.device,
                args=argparse.Namespace(**config.model_config.sg_former_model_args),
            )
            gnn_model = SGFormer(config, glo_gnn_model)
        model = MODEL_MAP[args.submodel](
            config,
            gnn_model,
            image_mlp_model,
            text_mlp_model,
            attribute_mlp_model,
            coordinator_mlp_model,
            combined_model,
        ).to(config.model_config.device)

    dataloaders = {
        "train": build_loaders(
            config,
            mode="train",
            dataset_class=DATA_LOADER_MAP[args.submodel],
        ),
        "valid": build_loaders(
            config,
            mode="valid",
            dataset_class=DATA_LOADER_MAP[args.submodel],
        ),
    }

    if args.mode == "train":
        # 设置损失函数和优化器
        criterion = CRITERION_MAP[args.submodel]
        optimizer = optim.SGD(
            model.parameters(), lr=config.model_config.lr, momentum=0.9, nesterov=True
        )
        # 训练模型
        TRAIN_FUNC_MAP[args.submodel](
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
        os.mkdir(config.dataset_config.predict_result_dir)

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
                loader = PREDICTOR_DATALOADER_MAP[args.dataset](config, page_id)
                collector = PREDICTOR_DATACOLLECTOR_MAP[args.submodel](
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
