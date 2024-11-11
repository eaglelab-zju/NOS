import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from UGNN.semi_heter.baselines.SIGN import preprocess as sign_preprocess
from utils import graph2edgeindex


class MLP(nn.Module):
    def __init__(
        self, config, input_dim, hidden_dims, output_dim, dropout, layer_norm=True
    ):
        super().__init__()

        self.config = config
        self.layers = nn.Sequential()
        if not hidden_dims:
            hidden_dims = []

        hidden_dims.insert(0, input_dim)
        hidden_dims.append(output_dim)
        for i in range(len(hidden_dims) - 1):
            if i < len(hidden_dims) - 2:
                if layer_norm:
                    self.layers.extend(
                        nn.Sequential(
                            nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=False),
                            nn.LayerNorm(hidden_dims[i + 1]),
                            nn.ReLU(),
                        )
                    )
                else:
                    self.layers.extend(
                        nn.Sequential(
                            nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=False),
                            nn.ReLU(),
                        )
                    )
            else:
                if dropout:
                    if layer_norm:
                        self.layers.extend(
                            nn.Sequential(
                                nn.Dropout(dropout),
                                nn.Linear(
                                    hidden_dims[i], hidden_dims[i + 1], bias=False
                                ),
                                nn.LayerNorm(hidden_dims[i + 1]),
                            )
                        )
                    else:
                        self.layers.extend(
                            nn.Sequential(
                                nn.Dropout(dropout),
                                nn.Linear(
                                    hidden_dims[i], hidden_dims[i + 1], bias=False
                                ),
                            )
                        )
                else:
                    if layer_norm:
                        self.layers.extend(
                            nn.Sequential(
                                nn.Linear(
                                    hidden_dims[i], hidden_dims[i + 1], bias=False
                                ),
                                nn.LayerNorm(hidden_dims[i + 1]),
                            )
                        )
                    else:
                        self.layers.extend(
                            nn.Sequential(
                                nn.Linear(
                                    hidden_dims[i], hidden_dims[i + 1], bias=False
                                ),
                            )
                        )

    def forward(self, x):
        return self.layers(x)


class MLPModel(nn.Module):
    def __init__(
        self,
        config,
        image_mlp_model,
        text_mlp_model,
        attribute_mlp_model,
        coordinator_mlp_model,
        classifier_model,
    ):
        super().__init__()
        self.config = config
        self.image_mlp_model = image_mlp_model
        self.text_mlp_model = text_mlp_model
        self.attribute_mlp_model = attribute_mlp_model
        self.coordinator_mlp_model = coordinator_mlp_model
        self.classifier_model = classifier_model

    def forward(self, batch):
        images, texts, attributes, coordinators = (
            batch["images"],
            batch["texts"],
            batch["attributes"],
            batch["coordinators"],
        )

        images_embeddings = self.image_mlp_model(images)
        texts_embeddings = self.text_mlp_model(texts)
        attributes_embeddings = self.attribute_mlp_model(attributes)
        coordinators_embeddings = self.coordinator_mlp_model(coordinators)

        combined_embeddings = torch.cat(
            (
                texts_embeddings,
                images_embeddings,
                attributes_embeddings,
                coordinators_embeddings,
            ),
            dim=-1,
        )
        return self.classifier_model(combined_embeddings)


class GCN(torch.nn.Module):
    def __init__(self, config, gcn_model):
        super().__init__()
        self.config = config
        self.gcn_model = gcn_model

    def forward(self, g, x):
        return self.gcn_model(graph=g, X=x)


class GraphSAGE(torch.nn.Module):
    def __init__(self, config, graph_sage_model):
        super().__init__()
        self.config = config
        self.graph_sage_model = graph_sage_model

    def forward(self, g, x):
        return self.graph_sage_model(graph=g, X=x)


class GAT(torch.nn.Module):
    def __init__(self, config, gat_model):
        super().__init__()
        self.config = config
        self.gat_model = gat_model

    def forward(self, g, x):
        return self.gat_model(graph=g, X=x)


class IGNN(torch.nn.Module):
    def __init__(self, config, ignn_model):
        super().__init__()
        self.config = config
        self.ignn_model = ignn_model

    def forward(self, g, x):
        x = self.ignn_model(graph=g, device=self.config.model_config.device, features=x)
        x = self.ignn_model.classifier(x)
        return x


class SIGN(torch.nn.Module):
    def __init__(self, config, sign_model):
        super().__init__()
        self.config = config
        self.sign_model = sign_model

    def forward(self, g, x):
        x = sign_preprocess(g, x, R=self.config.model_config.sign_model_args["n_hops"])
        x = self.sign_model(x)
        return x


class OrderedGNN(torch.nn.Module):
    def __init__(self, config, ordered_gnn_model):
        super().__init__()
        self.config = config
        self.ordered_gnn_model = ordered_gnn_model

    def forward(self, g, x):
        edge_index = torch.tensor(
            np.array(graph2edgeindex(g)),
            device=self.config.model_config.device,
            dtype=torch.long,
        )
        x = self.ordered_gnn_model(x=x, edge_index=edge_index)
        return x


class SGFormer(torch.nn.Module):
    def __init__(self, config, sg_former_model):
        super().__init__()
        self.config = config
        self.sg_former_model = sg_former_model

    def forward(self, g, x):
        edge_index = torch.tensor(
            np.array(graph2edgeindex(g)),
            device=self.config.model_config.device,
            dtype=torch.long,
        )
        x = self.sg_former_model(x=x, edge_index=edge_index)
        return x


class GNNModel(torch.nn.Module):
    def __init__(
        self,
        config,
        gnn_model,
        image_mlp_model,
        text_mlp_model,
        attribute_mlp_model,
        coordinator_mlp_model,
        combined_model,
    ):
        super().__init__()
        self.config = config
        self.gnn_model = gnn_model
        self.image_mlp_model = image_mlp_model
        self.text_mlp_model = text_mlp_model
        self.attribute_mlp_model = attribute_mlp_model
        self.coordinator_mlp_model = coordinator_mlp_model
        self.combined_model = combined_model
        self.layer_norm = nn.LayerNorm(config.model_config.gnn_input_dim)

    def forward(self, graph):
        images, texts, attributes, coordinators = (
            graph.ndata["images"],
            graph.ndata["texts"],
            graph.ndata["attributes"],
            graph.ndata["coordinators"],
        )
        images_embeddings = self.image_mlp_model(images)
        texts_embeddings = self.text_mlp_model(texts)
        attributes_embeddings = self.attribute_mlp_model(attributes)
        coordinators_embeddings = self.coordinator_mlp_model(coordinators)
        combined_embeddings = torch.cat(
            (
                images_embeddings,
                texts_embeddings,
                attributes_embeddings,
                coordinators_embeddings,
            ),
            dim=-1,
        )
        combined_embeddings = self.combined_model(combined_embeddings)
        output = self.gnn_model(graph, combined_embeddings)
        return output


class CrossAttentionEncoderLayer(torch.nn.Module):
    def __init__(self, config, model_dim, num_heads, dim_feedforward, dropout):
        super().__init__()
        self.config = config

        # 图像的自注意力
        self.image_self_attn = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout
        )
        # 图像对文本的交叉注意力
        self.image_cross_attn = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout
        )
        # 图像的前馈网络
        self.image_ffn = nn.Sequential(
            nn.Linear(model_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, model_dim),
            nn.Dropout(dropout),
        )
        # 图像的层归一化
        self.image_norm1 = nn.LayerNorm(model_dim)
        self.image_norm2 = nn.LayerNorm(model_dim)
        self.image_norm3 = nn.LayerNorm(model_dim)

        # 文本的自注意力
        self.text_self_attn = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout
        )
        # 文本对图像的交叉注意力
        self.text_cross_attn = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout
        )
        # 文本的前馈网络
        self.text_ffn = nn.Sequential(
            nn.Linear(model_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, model_dim),
            nn.Dropout(dropout),
        )
        # 文本的层归一化
        self.text_norm1 = nn.LayerNorm(model_dim)
        self.text_norm2 = nn.LayerNorm(model_dim)
        self.text_norm3 = nn.LayerNorm(model_dim)

    def forward(self, image_features, text_features, image_mask, text_mask):
        """
        参数:
            image_features: [seq_len_image, batch_size, model_dim]
            text_features: [seq_len_text, batch_size, model_dim]
            image_mask: 可选的图像掩码
            text_mask: 可选的文本掩码
        返回:
            更新后的图像和文本特征
        """
        # 图像的自注意力
        image_attn_output, _ = self.image_self_attn(
            image_features, image_features, image_features, key_padding_mask=image_mask
        )
        image_features = self.image_norm1(image_features + image_attn_output)

        # 图像对文本的交叉注意力
        image_attn_output, _ = self.image_cross_attn(
            image_features, text_features, text_features, key_padding_mask=text_mask
        )
        image_features = self.image_norm2(image_features + image_attn_output)

        # 图像的前馈网络
        image_ffn_output = self.image_ffn(image_features)
        image_features = self.image_norm3(image_features + image_ffn_output)

        # 文本的自注意力
        text_attn_output, _ = self.text_self_attn(
            text_features, text_features, text_features, key_padding_mask=text_mask
        )
        text_features = self.text_norm1(text_features + text_attn_output)

        # 文本对图像的交叉注意力
        text_attn_output, _ = self.text_cross_attn(
            text_features, image_features, image_features, key_padding_mask=image_mask
        )
        text_features = self.text_norm2(text_features + text_attn_output)

        # 文本的前馈网络
        text_ffn_output = self.text_ffn(text_features)
        text_features = self.text_norm3(text_features + text_ffn_output)

        return image_features, text_features


class CrossAttentionEncoderModel(nn.Module):
    def __init__(
        self, config, model_dim, num_heads, num_layers, dim_feedforward, dropout
    ):
        super().__init__()
        self.config = config
        # 堆叠多个并行 Transformer Decoder 层
        self.layers = nn.ModuleList(
            [
                CrossAttentionEncoderLayer(
                    config, model_dim, num_heads, dim_feedforward, dropout
                )
                for _ in range(num_layers)
            ]
        )
        # 最后的层归一化
        self.norm_image = nn.LayerNorm(model_dim)
        self.norm_text = nn.LayerNorm(model_dim)

    def forward(self, image_features, text_features, image_mask=None, text_mask=None):
        """
        参数:
            image_features: [batch_size, seq_len_image, image_dim]
            text_features: [batch_size, seq_len_text, text_dim]
            image_mask: 可选的图像掩码
            text_mask: 可选的文本掩码
        返回:
            更新后的图像和文本特征
        """
        # 转换为 [seq_len, batch_size, model_dim] 以适应 nn.MultiheadAttention 的输入格式
        image_features = image_features.permute(1, 0, 2)
        text_features = text_features.permute(1, 0, 2)

        for layer in self.layers:
            image_features, text_features = layer(
                image_features, text_features, image_mask, text_mask
            )

        # 转回 [batch_size, seq_len, model_dim]
        image_features = image_features.permute(1, 0, 2)
        text_features = text_features.permute(1, 0, 2)

        # 最后的层归一化
        image_features = self.norm_image(image_features)
        text_features = self.norm_text(text_features)

        return image_features, text_features


class TransformerModel(torch.nn.Module):
    def __init__(
        self,
        config,
        text_mlp_model,
        attributes_model,
        coordinator_model,
        transformer_decoder_model,
        classifier_model,
    ):
        super().__init__()
        self.config = config
        # MLP
        self.text_mlp_model = text_mlp_model
        # MLP
        self.attributes_model = attributes_model
        # Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding
        self.coordinator_model = coordinator_model
        # Transformer Parallel Decoder(cross-attention)
        self.transformer_decoder_model = transformer_decoder_model
        # MLP
        self.classifier_model = classifier_model

    def forward(self, nodes_batch):
        # [batch_size, seq_length, channel_num, image_size, image_size]
        images_embeddings = nodes_batch["images"]
        texts = nodes_batch["texts"]
        coordinators = nodes_batch["coordinators"]
        attributes = nodes_batch["attributes"]
        attn_masks = nodes_batch["attn_masks"]

        texts_embeddings = self.text_mlp_model(texts)
        attributes_embeddings = self.attributes_model(attributes)
        coordinators_embeddings = self.coordinator_model(coordinators)

        texts_embeddings = torch.cat(
            (texts_embeddings, attributes_embeddings, coordinators_embeddings), dim=-1
        )
        images_embeddings, texts_embeddings = self.transformer_decoder_model(
            images_embeddings, texts_embeddings, attn_masks, attn_masks
        )

        # [batch_size, model_dim * 2]
        final_embeddings = torch.cat((images_embeddings, texts_embeddings), dim=-1)
        # TODO: seq_length 维度可以保留，算 loss 的时候再处理
        final_embeddings = torch.flatten(final_embeddings, start_dim=0, end_dim=1)
        # 分类头
        logits = self.classifier_model(final_embeddings)  # [batch_size, num_classes]
        # return torch.softmax(logits, dim=-1)
        return logits


class DETRModel(torch.nn.Module):
    def __init__(
        self,
        config,
        text_mlp_model,
        attributes_model,
        coordinator_model,
        node_mlp_model,
        detr_model,
        classifier_model,
    ):
        super().__init__()
        self.config = config
        # MLP
        self.text_mlp_model = text_mlp_model
        # MLP ?
        self.attributes_model = attributes_model
        # Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding
        self.coordinator_model = coordinator_model
        # MLP
        self.node_mlp_model = node_mlp_model
        # Transformer Parallel Decoder(cross-attention)
        self.detr_model = detr_model
        # MLP
        self.classifier_model = classifier_model

    def forward(self, nodes_batch):
        # [batch_size, channel_num, image_size, image_size]
        images = nodes_batch["images"]
        # [batch_size, seq_length, channel_num, image_size, image_size]
        texts_embeddings = nodes_batch["texts"]
        coordinators = nodes_batch["coordinators"]
        attributes = nodes_batch["attributes"]

        texts_embeddings = self.text_mlp_model(texts_embeddings)
        attributes_embeddings = self.attributes_model(attributes)
        coordinators_embeddings = self.coordinator_model(coordinators)

        other_embeddings = torch.cat(
            (texts_embeddings, attributes_embeddings, coordinators_embeddings), dim=-1
        )
        other_embeddings = self.node_mlp_model(other_embeddings)

        return self.detr_model(
            images,
            other_embeddings,
        )
