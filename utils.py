import torch
from torch import Tensor
from torch_geometric.data import Data
from torch import LongTensor
from itertools import product


def complete_edge_index(n: int) -> LongTensor:
    edges = list(filter(lambda e: e[0] != e[1], product(range(n), range(n))))
    return torch.tensor(edges, dtype=torch.long).T


def add_complete_graph_edge_index(data: Data) -> Data:
    data.complete_edge_index = complete_edge_index(data.num_nodes)
    return data


def total_absolute_error(pred: Tensor, target: Tensor, batch_dim: int = 0) -> Tensor:
    return (pred - target).abs().sum(batch_dim)
