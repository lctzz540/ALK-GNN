import torch
import torch.nn as nn
from typing import Optional
import torch_geometric.nn as geom_nn
from torch import Tensor

from dataset import MoleculeDataset


class NaiveEuclideanGNN(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_layers: int,
        num_spatial_dims: int,
        final_embedding_size: Optional[int] = None,
        act: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()
        self.f_initial_embed = nn.Embedding(100, hidden_channels)
        self.f_pos_embed = nn.Linear(num_spatial_dims, hidden_channels)
        self.f_combine = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels), act
        )

        if final_embedding_size is None:
            final_embedding_size = hidden_channels

        self.gnn = geom_nn.models.GIN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=final_embedding_size,
            num_layers=num_layers,
            act=act,
        )

        self.aggregation = geom_nn.aggr.SumAggregation()
        self.f_predict = nn.Sequential(
            nn.Linear(final_embedding_size, final_embedding_size),
            act,
            nn.Linear(final_embedding_size, 1),
        )

    def encode(self, data: MoleculeDataset) -> Tensor:
        atom_embedding = self.f_initial_embed(data.z)
        pos_embedding = self.f_pos_embed(data.pos)

        initial_node_embed = self.f_combine(
            torch.cat((atom_embedding, pos_embedding), dim=-1)
        )

        node_embed = self.gnn(initial_node_embed, data.edge_index)
        return node_embed

    def forward(self, data: MoleculeDataset) -> Tensor:
        node_embed = self.encode(data)
        aggr = self.aggregation(node_embed, data.batch)
        return self.f_predict(aggr)
