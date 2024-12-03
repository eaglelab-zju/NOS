import json
import time

import torch


class ModelConfig:
    pass


class DatasetConfig:
    def __init__(self, split):
        self.name = "abstract"
        self.split = split

        self.image_transform_resize = (224, 224)
        self.clickable_noise_ratio = 0.4  # clickable 噪声比例
        self._load_dataset_dir()
        self._load_data()

        print(
            f"train_dataset: {self.train_data_ids}, valid_dataset: {self.valid_data_ids}"
        )

    def _load_dataset_dir(self):
        raise NotImplementedError()

    def adjust(self, model_config: ModelConfig):
        return

    def _load_data(self):
        with open(
            f"{self.dataset_dir}/dataset_split_{self.split}.json", "r", encoding="utf-8"
        ) as f:
            dataset = json.load(f)
        self.train_data_ids = dataset["train"]
        self.valid_data_ids = dataset["valid"]
        self.test_data_ids = dataset["test"]

    def print(self):
        print(self.__dict__)


class NOSRawDatasetConfig(DatasetConfig):
    def __init__(self, split):
        super().__init__(split)
        self.border = 0
        self.phone_height, self.phone_width = 1600, 720
        self.extra_top, self.extra_bottom = 45, 78
        self.length_threshold = 5
        self.image_height, self.image_width = (
            self.phone_height - self.extra_top - self.extra_bottom,
            self.phone_width,
        )

    def _load_dataset_dir(self):
        self.name = "nos-raw-labeled"
        self.dataset_dir = f"./dataset/{self.name}"
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
        self.name = "rico-labeled"
        self.dataset_dir = f"./dataset/{self.name}"
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
        self.rico_phone_height, self.rico_phone_width = 2560, 1440
        self.rico_extra_bottom = 168
        self.rico_length_threshold = 5

        self.nos_raw_phone_height, self.nos_raw_phone_width = 1600, 720
        self.nos_raw_extra_bottom = 78
        self.nos_raw_length_threshold = 5

    def _load_dataset_dir(self):
        self.name = "mixed"
        self.dataset_dir = f"./dataset/{self.name}"
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
        self.num_workers = 4

        self.batch_size = 64
        self.accumulation_steps = 0
        self.num_epochs = 100
        self.early_stopping = 100

    def adjust(self, dataset_config: DatasetConfig):
        return

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

        self.coordinate_mlp_input_dim = 4
        self.coordinate_mlp_hidden_dims = [128]
        self.coordinate_mlp_output_dim = 256
        self.coordinate_mlp_dropout = 0.5

        self.use_blation = False
        if self.use_blation:
            self.combined_mlp_input_dim = 768
        else:
            self.combined_mlp_input_dim = (
                self.image_mlp_output_dim
                + self.text_mlp_output_dim
                + self.attribute_mlp_output_dim
                + self.coordinate_mlp_output_dim
            )
        self.combined_mlp_hidden_dims = [512]
        self.combined_mlp_output_dim = 256
        self.combined_mlp_dropout = 0.5

        self.gnn_input_dim = self.combined_mlp_output_dim
        self.gnn_output_dim = 2

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
        self.ignn_model_loss = "ce"

        ## 以下参数没有实际作用
        self.ignn_model_n_epochs = None
        self.ignn_model_lr = 1e-3
        self.ignn_model_l2_coef = 0.2
        self.ignn_model_early_stop = None
        self.ignn_model_lda = None
        ## 以上参数没有实际作用

        self.classifier_model_output_dim = self.gnn_output_dim

    def adjust(self, dataset_config: DatasetConfig):
        if "rico" in dataset_config.name or "mixed" in dataset_config.name:
            self.lr = 1e-4

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

        self.dataset_config.predict_result_dir = f"./predict_result/{self.model_config.name}/{self.dataset_config.name}_split_{self.dataset_config.split}"
        self.checkpoint_dir = f"./model_checkpoint/{self.model_config.name}/{self.dataset_config.name}_split_{self.dataset_config.split}"
        if args.mode == "predict":
            self.checkpoint_file = f"{self.checkpoint_dir}/{args.checkpoint}.pt"
        else:
            self.checkpoint_file = f"{self.checkpoint_dir}/{time.time()}.pt"

    def done(self):
        print(self.__dict__)
        self.dataset_config.print()
        self.model_config.print()
        print("[Init] Config init finished")
