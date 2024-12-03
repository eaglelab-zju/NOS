import numpy as np
import torch
import torch.nn as nn
from torchvision import models


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


class IGNN(torch.nn.Module):
    def __init__(self, config, ignn_model):
        super().__init__()
        self.config = config
        self.ignn_model = ignn_model

    def forward(self, g, x):
        x = self.ignn_model(graph=g, device=self.config.model_config.device, features=x)
        x = self.ignn_model.classifier(x)
        return x


class GNNModel(torch.nn.Module):
    def __init__(
        self,
        config,
        gnn_model,
        image_mlp_model,
        text_mlp_model,
        attribute_mlp_model,
        coordinate_mlp_model,
        combined_model,
    ):
        super().__init__()
        self.config = config
        self.gnn_model = gnn_model
        self.image_mlp_model = image_mlp_model
        self.text_mlp_model = text_mlp_model
        self.attribute_mlp_model = attribute_mlp_model
        self.coordinator_mlp_model = coordinate_mlp_model
        self.combined_model = combined_model
        self.layer_norm = nn.LayerNorm(config.model_config.gnn_input_dim)

    def forward(self, graph):
        images, texts, attributes, coordinates = (
            graph.ndata["images"],
            graph.ndata["texts"],
            graph.ndata["attributes"],
            graph.ndata["coordinates"],
        )
        images_embeddings = self.image_mlp_model(images)
        texts_embeddings = self.text_mlp_model(texts)
        attributes_embeddings = self.attribute_mlp_model(attributes)
        coordinates_embeddings = self.coordinator_mlp_model(coordinates)
        combined_embeddings = torch.cat(
            (
                images_embeddings,
                texts_embeddings,
                attributes_embeddings,
                coordinates_embeddings,
            ),
            dim=-1,
        )
        combined_embeddings = self.combined_model(combined_embeddings)
        output = self.gnn_model(graph, combined_embeddings)
        return output
