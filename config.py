import json
import time

import torch


class ModelConfig:
    pass


class DatasetConfig:
    def __init__(self, split):
        self.name = "abstract"
        self.clickable_noise_ratio = 0.4

        self.down_sampling = False
        self.down_sampling_ratio = 0.8
        self.split = split

        self.image_transform_resize = (224, 224)
        self.graph_node_num_thresold = 100000

        self._load_dataset_dir()
        self._load_data()

        print(
            f"train_dataset: {self.train_data_ids}, valid_dataset: {self.valid_data_ids}"
        )

        self.predict_result_dir = "./predict_result"

    def _load_dataset_dir(self):
        raise NotImplementedError()

    def _load_data(self):
        self.manual_weight = 1
        self.clickable_weight = 1

        with open(
            f"{self.dataset_dir}/dataset_split_{self.split}.json", "r", encoding="utf-8"
        ) as f:
            dataset = json.load(f)
        self.train_data_ids = dataset["train"]
        self.valid_data_ids = dataset["valid"]
        self.test_data_ids = dataset["test"]
        self.labelled_data_id_threshold = 100000

    def adjust(self, model_config: ModelConfig):
        if "rico" in self.name:
            model_config.version = self.name
        else:
            model_config.version = "a11y_random_1"
        self.train_data_file = f"{self.dataset_dir}/{model_config.name}_{model_config.version}_train_data.json"
        self.valid_data_file = f"{self.dataset_dir}/{model_config.name}_{model_config.version}_valid_data.json"
        self.test_data_file = f"{self.dataset_dir}/{model_config.name}_{model_config.version}_test_data.json"

    def print(self):
        print(self.__dict__)


class A11yDatasetConfig(DatasetConfig):
    def __init__(self, split):
        super().__init__(split)
        self.classifier_positive_threshold = 0.3

        self.border = 0
        self.phone_height, self.phone_width = 1600, 720
        self.extra_top, self.extra_bottom = 45, 78
        self.length_threshold = 5
        self.image_height, self.image_width = (
            self.phone_height - self.extra_top - self.extra_bottom,
            self.phone_width,
        )

    def _load_dataset_dir(self):
        self.name = "a11y_final_v3"
        self.dataset_dir = f"../../my_dataset/{self.name}"
        self.image_dir = f"{self.dataset_dir}/screenshot"
        self.hierarchy_dir = f"{self.dataset_dir}/hierarchy"
        self.graph_dir = f"{self.dataset_dir}/graph"
        self.box_image_dir = f"{self.dataset_dir}/box"
        self.text_feat_dir = f"{self.dataset_dir}/text_feat"
        self.box_feat_dir = f"{self.dataset_dir}/box_feat"

    def print(self):
        print(self.__dict__)


class RicoDatasetConfig(DatasetConfig):
    def __init__(self, split):
        super().__init__(split)
        self.border = 0
        self.phone_height, self.phone_width = 2560, 1440
        self.extra_top, self.extra_bottom = 84, 168
        self.length_threshold = 5
        self.image_height, self.image_width = (
            self.phone_height - self.extra_top - self.extra_bottom,
            self.phone_width,
        )

    def _load_dataset_dir(self):
        self.name = "rico_final_v2"
        self.dataset_dir = f"../../my_dataset/{self.name}"
        self.image_dir = f"{self.dataset_dir}/screenshot"
        self.hierarchy_dir = f"{self.dataset_dir}/hierarchy"
        self.graph_dir = f"{self.dataset_dir}/graph"
        self.box_image_dir = f"{self.dataset_dir}/box"
        self.text_feat_dir = f"{self.dataset_dir}/text_feat"
        self.box_feat_dir = f"{self.dataset_dir}/box_feat"

    def print(self):
        print(self.__dict__)


class MixedDatasetConfig(DatasetConfig):
    def __init__(self, split):
        super().__init__(split)
        self.rico_border = 0
        self.rico_phone_height, self.rico_phone_width = 2560, 1440
        self.rico_extra_top, self.rico_extra_bottom = 84, 168
        self.rico_length_threshold = 5
        self.rico_image_height, self.rico_image_width = (
            self.rico_phone_height - self.rico_extra_top - self.rico_extra_bottom,
            self.rico_phone_width,
        )

        self.a11y_border = 0
        self.a11y_phone_height, self.a11y_phone_width = 1600, 720
        self.a11y_extra_top, self.a11y_extra_bottom = 45, 78
        self.a11y_length_threshold = 5
        self.a11y_image_height, self.a11y_image_width = (
            self.a11y_phone_height - self.a11y_extra_top - self.a11y_extra_bottom,
            self.a11y_phone_width,
        )

    def _load_dataset_dir(self):
        self.name = "mixed_final_v1"
        self.dataset_dir = f"../../my_dataset/{self.name}"
        self.image_dir = f"{self.dataset_dir}/screenshot"
        self.hierarchy_dir = f"{self.dataset_dir}/hierarchy"
        self.graph_dir = f"{self.dataset_dir}/graph"
        self.box_image_dir = f"{self.dataset_dir}/box"
        self.text_feat_dir = f"{self.dataset_dir}/text_feat"
        self.box_feat_dir = f"{self.dataset_dir}/box_feat"

    def print(self):
        print(self.__dict__)


class ModelConfig:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = "abstract"
        self.version = "abstract"
        self.num_workers = 4

        self.batch_size = 128 * 6
        self.accumulation_steps = 0
        self.num_epochs = 100
        self.early_stopping = 100

    def adjust(self, dataset_config: DatasetConfig):
        raise NotImplementedError()

    def print(self):
        print(self.__dict__)


class MLPModelConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        self.name = "mlp"
        self.num_epochs = 5000
        self.lr = 0.05

        self.image_dim = 1000
        self.text_dim = 768

        self.image_mlp_input_dim = self.image_dim
        self.image_mlp_hidden_dims = [512]
        self.image_mlp_output_dim = 256
        self.image_mlp_dropout = 0.5

        self.text_mlp_input_dim = self.text_dim
        self.text_mlp_hidden_dims = [512]
        self.text_mlp_output_dim = 256
        self.text_mlp_dropout = 0.5

        self.attribute_mlp_input_dim = 11
        self.attribute_mlp_hidden_dims = [128]
        self.attribute_mlp_output_dim = 256
        self.attribute_mlp_dropout = 0.5

        self.coordinator_mlp_input_dim = 4
        self.coordinator_mlp_hidden_dims = [128]
        self.coordinator_mlp_output_dim = 256
        self.coordinator_mlp_dropout = 0.5

        self.classifier_mlp_input_dim = (
            self.image_mlp_output_dim
            + self.text_mlp_output_dim
            + self.attribute_mlp_output_dim
            + self.coordinator_mlp_output_dim
        )
        self.classifier_mlp_hidden_dims = [512, 256, 32]
        self.classifier_mlp_output_dim = 2
        self.classifier_mlp_dropout = 0.5

    def adjust(self, dataset_config: DatasetConfig):
        pass

    def print(self):
        print(self.__dict__)


class GNNModelConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        self.num_workers = 4
        self.accumulation_steps = 64

        self.name = "gnn"
        self.num_epochs = 5000
        self.lr = 1e-3
        self.batch_size = 1

        self.image_dim = 1000
        self.text_dim = 768

        self.image_mlp_input_dim = self.image_dim
        self.image_mlp_hidden_dims = [512]
        self.image_mlp_output_dim = 256
        self.image_mlp_dropout = 0.5

        self.text_mlp_input_dim = self.text_dim
        self.text_mlp_hidden_dims = [512]
        self.text_mlp_output_dim = 256
        self.text_mlp_dropout = 0.5

        self.attribute_mlp_input_dim = 11
        self.attribute_mlp_hidden_dims = [128]
        self.attribute_mlp_output_dim = 256
        self.attribute_mlp_dropout = 0.5

        self.coordinator_mlp_input_dim = 4
        self.coordinator_mlp_hidden_dims = [128]
        self.coordinator_mlp_output_dim = 256
        self.coordinator_mlp_dropout = 0.5

        self.use_blation = False
        if self.use_blation:
            self.combined_mlp_input_dim = 768
        else:
            self.combined_mlp_input_dim = (
                self.image_mlp_output_dim
                + self.text_mlp_output_dim
                + self.attribute_mlp_output_dim
                + self.coordinator_mlp_output_dim
            )
        self.combined_mlp_hidden_dims = [512]
        self.combined_mlp_output_dim = 256
        self.combined_mlp_dropout = 0.5

        self.gnn_input_dim = self.combined_mlp_output_dim
        self.gnn_output_dim = 2

    def adjust(self, dataset_config: DatasetConfig):
        pass

    def print(self):
        print(self.__dict__)


class GCNModelConfig(GNNModelConfig):
    def __init__(self):
        super().__init__()
        self.name = "gcn"

        self.gcn_model_in_features = self.gnn_input_dim
        self.gcn_model_class_num = self.gnn_output_dim
        self.gcn_model_args = {
            "lr": None,
            "l2_coef": None,
            "epochs": None,
            "patience": None,
            "layers": 2,
            "dropout": 0.2,
            "nhidden": 128,
        }

    def adjust(self, dataset_config: DatasetConfig):
        super().adjust(dataset_config)
        if "rico" in dataset_config.name or "mixed" in dataset_config.name:
            self.lr = 1e-4

    def print(self):
        print(self.__dict__)


class GATModelConfig(GNNModelConfig):
    def __init__(self):
        super().__init__()
        self.name = "gat"

        self.gat_model_in_features = self.gnn_input_dim
        self.gat_model_class_num = self.gnn_output_dim
        self.gat_model_args = {
            "lr": None,
            "l2_coef": None,
            "epochs": None,
            "patience": None,
            "layers": 2,
            "dropout": 0.2,
            "nhidden": 128,
        }

    def print(self):
        print(self.__dict__)


class GraphSAGEModelConfig(GNNModelConfig):
    def __init__(self):
        super().__init__()
        self.name = "sage"
        self.lr = 1e-4

        self.graph_sage_model_in_features = self.gnn_input_dim
        self.graph_sage_model_class_num = self.gnn_output_dim
        self.graph_sage_model_args = {
            "lr": None,
            "l2_coef": None,
            "epochs": None,
            "patience": None,
            "layers": 2,
            "dropout": 0.2,
            "nhidden": 128,
        }

    def print(self):
        print(self.__dict__)


class IGNNModelConfig(GNNModelConfig):
    def __init__(self):
        super().__init__()
        self.name = "ignn"

        # 输入特征维度
        self.ignn_model_in_feats = self.gnn_input_dim
        # 表征维度
        self.ignn_model_h_feats = 128
        # 使用的邻域跳数
        self.ignn_model_n_hops = 10  # 10
        # 输入特征 dropout 比例，输入维度大时建议调大
        self.ignn_model_nas_dropout = 0.2
        # 隐层特征 dropout 比例，输入维度大时建议调大
        self.ignn_model_nss_dropout = 0.2
        # 隐层特征 dropout 比例，输入维度大时建议调大
        self.ignn_model_clf_dropout = 0.2
        # 无用，可忽略
        self.ignn_model_n_intervals = 3
        # 无用，可忽略
        self.ignn_model_out_ndim_trans = 128
        # 不用改
        self.ignn_model_nie = "gcn-nnie-nst"
        # 不用改
        self.ignn_model_nrl = "concat"
        # 不用改
        self.ignn_model_act = "relu"
        # True / False 都可以尝试
        self.ignn_model_layer_norm = True
        # 不用改
        self.ignn_model_n_nodes = None
        # 无用，可忽略
        self.ignn_model_ndim_h_a = 64
        # 无用，可忽略
        self.ignn_model_num_heads = 1
        # 无用，可忽略
        self.ignn_model_transform_first = False
        # 无用，可忽略
        self.ignn_model_trans_layer_num = 5
        # IGNN 层数，一般图一层即可，链式图多层效果好
        # Tree 图可以将 n_hops 设成 1，调大该参数
        self.ignn_model_ignn_layer_num = 1  # 1
        # 必须是 True
        self.ignn_model_no_save = True
        self.classifier_model_output_dim = self.gnn_output_dim

    def adjust(self, dataset_config: DatasetConfig):
        super().adjust(dataset_config)
        if "rico" in dataset_config.name or "mixed" in dataset_config.name:
            self.lr = 1e-4

    def print(self):
        print(self.__dict__)


class SIGNModelConfig(GNNModelConfig):
    def __init__(self):
        super().__init__()
        self.name = "sign"
        self.lr = 1e-4

        self.sign_model_in_features = self.gnn_input_dim
        self.sign_model_class_num = self.gnn_output_dim
        self.sign_model_args = {
            "nhidden": 128,
            "n_hops": 10,
            # 每跳 MLP 层数
            "n_layers": 2,
            # 跳数多就调大
            "dropout": 0.8,
            "lr": None,
            "l2_coef": None,
            "epochs": None,
            "patience": None,
        }

    def print(self):
        print(self.__dict__)


class OrderedGNNModelConfig(GNNModelConfig):
    def __init__(self):
        super().__init__()
        self.name = "ordered_gnn"
        self.lr = 1e-4

        self.ordered_gnn_model_in_features = self.gnn_input_dim
        self.ordered_gnn_model_class_num = self.gnn_output_dim
        self.ordered_gnn_model_args = {
            "hidden_channel": 128,
            "num_layers": 2,
            # 输入特征 MLP 层数
            "num_layers_input": 2,
            "add_self_loops": False,
            # 是否所有跳使用同一 Attention 机制，一般为 False
            "global_gating": False,
            # 默认 False 即可
            "simple_gating": False,
            # 默认即可
            "tm": True,
            # 默认即可
            "diff_or": True,
            # 输入特征大，就调大
            "dropout": 0.2,
            # 输入特征大，就调大
            "dropout2": 0.2,
            # 默认 64
            "chunk_size": 64,
            "lr": None,
            "l2_coef": None,
            "epochs": None,
            "patience": None,
        }

    def print(self):
        print(self.__dict__)


class SGFormerModelConfig(GNNModelConfig):
    def __init__(self):
        super().__init__()
        self.name = "sg_former"

        self.sg_former_model_in_features = self.gnn_input_dim
        self.sg_former_model_class_num = self.gnn_output_dim
        self.sg_former_model_args = {
            "nhidden": 128,
            "num_layers": 2,
            "num_heads": 4,
            # 默认即可
            "alpha": 0.5,
            # 输入特征大，就调大
            "dropout": 0.2,
            # 默认即可
            "use_bn": True,
            # 默认即可
            "use_residual": True,
            # 默认即可
            "use_weight": True,
            # 默认即可
            "use_graph": True,
            # 默认即可
            "use_act": False,
            # 图特征融合比例
            "graph_weight": 0.4,
            # 默认即可
            "gnn": "gcn",
            # 默认即可
            "aggregate": "add",
            "lr": None,
            "l2_coef": None,
            "epochs": None,
            "patience": None,
        }

    def print(self):
        print(self.__dict__)


class TransformerModelConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        self.name = "transformer"
        self.num_epochs = 5000
        self.accumulation_steps = 16
        self.batch_size = 1

        self.lr = 1e-3
        # 组件序列长度
        self.seq_length = 128
        # 前后重叠窗口大小
        self.seq_overlap = 16

        # vit
        self.image_pretrained_model_output_dim = 1000
        # bert
        self.text_pretrained_model_output_dim = 768
        # mlp
        self.text_mlp_model_dropout = 0.5
        self.text_mlp_model_input_dim = self.text_pretrained_model_output_dim
        self.text_mlp_model_hidden_dims = [512]
        self.text_mlp_model_output_dim = 256

        # mlp?
        self.attributes_model_dropout = 0.5
        self.attributes_model_input_dim = 11
        self.attributes_model_hidden_dims = [256]
        self.attributes_model_output_dim = 488

        # mlp / position encoder
        self.coordinator_model_dropout = 0.5
        self.coordinator_model_input_dim = 4
        self.coordinator_model_hidden_dims = [128]
        self.coordinator_model_output_dim = 256

        self.text_attributes_coordinator_concat_dim = (
            self.text_mlp_model_output_dim
            + self.attributes_model_output_dim
            + self.coordinator_model_output_dim
        )

        # mlp
        self.node_mlp_model_dropout = 0.5
        self.node_mlp_model_input_dim = self.text_attributes_coordinator_concat_dim
        self.node_mlp_model_hidden_dims = [512]
        self.node_mlp_model_output_dim = 256

        self.text_attributes_coordinator_concat_dim = self.node_mlp_model_output_dim

        # transformer decoder
        self.transformer_decoder_model_layer_input_dim = (
            self.image_pretrained_model_output_dim
        )
        self.transformer_decoder_model_num_heads = 8
        self.transformer_decoder_model_num_layers = 6
        self.transformer_decoder_model_dim_feedforward = 1024
        self.transformer_decoder_model_dropout = 0.2
        # layer(the same as model)
        self.transformer_decoder_layer_input_dim = (
            self.image_pretrained_model_output_dim
        )
        self.transformer_decoder_layer_num_heads = 8
        self.transformer_decoder_layer_dim_feedforward = 1024
        self.transformer_decoder_layer_dropout = 0.2

        # mlp
        self.classifier_model_dropout = 0.5
        self.classifier_model_input_dim = self.image_pretrained_model_output_dim * 2
        self.classifier_model_hidden_dims = [512, 64]
        self.classifier_model_output_dim = 2

    def adjust(self, dataset_config: DatasetConfig):
        pass

    def print(self):
        print(self.__dict__)


class DETRModelConfig(TransformerModelConfig):
    def __init__(self):
        super().__init__()
        self.name = "detr"
        self.accumulation_steps = 0
        self.batch_size = 128

        self.detr_args = {
            "num_classes": 1,
            "num_queries": self.seq_length,
            "backbone": "resnet50",
            "lr_backbone": 1e-5,
            "masks": False,
            "dilation": True,
            "position_embedding": "sine",
            "hidden_dim": 256,
            "dropout": 0.1,
            "nheads": 8,
            "dim_feedforward": 2048,
            "enc_layers": 6,
            "dec_layers": 6,
            "pre_norm": True,
        }

    def adjust(self, dataset_config: DatasetConfig):
        pass

    def print(self):
        print(self.__dict__)


class Config:
    def load_config(
        self, args, model_config: ModelConfig, dataset_config: DatasetConfig
    ):
        self.args = args
        self.mode = args.mode
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.model_config.adjust(self.dataset_config)
        self.dataset_config.adjust(self.model_config)

        self.dataset_config.predict_result_dir = f"{self.dataset_config.predict_result_dir}/{self.args.submodel}/{self.model_config.name}_{self.model_config.version}"
        if args.mode == "predict":
            self.checkpoint_file = f"../../model_checkpoint/{self.args.submodel}/{self.dataset_config.name}_{self.model_config.version}_{args.checkpoint}.pt"
        else:
            self.checkpoint_file = f"../../model_checkpoint/{self.args.submodel}/{self.dataset_config.name}_{self.model_config.version}_{time.time()}.pt"

    def done(self):
        print(
            f"[Init] Config init finished, mode: {self.mode}, dataset: {self.dataset_config.name}, model: {self.model_config.name}, device: {self.model_config.device}, checkpoint: {self.checkpoint_file}, batch_size: {self.model_config.batch_size}"
        )
        print(self.__dict__)
        self.dataset_config.print()
        self.model_config.print()
