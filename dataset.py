from torch_geometric.data import Dataset, Data
import torch
import os
import pandas as pd
from tqdm import tqdm
import deepchem as dc
from rdkit import Chem
import numpy as np


class MoleculeDataset(Dataset):
    def __init__(self, root, filename, transform=None, pre_transform=None):
        self.root = root
        self.filename = filename
        self.data = self.get()
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

        data = Data()

        node_features_list = []
        edge_index_list = []
        edge_features_list = []
        label_list = []

        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            mol = Chem.MolFromSmiles(row["Canomicalsmiles"])
            f = featurizer._featurize(mol)
            node_features = torch.tensor(f.node_features, dtype=torch.float)
            edge_index = torch.tensor(f.edge_index, dtype=torch.int64)
            edge_features = torch.tensor(f.edge_features, dtype=torch.float)

            label = self._get_labels(row["pChEMBL"])

            node_features_list.append(node_features)
            edge_index_list.append(edge_index)
            edge_features_list.append(edge_features)
            label_list.append(label)

        data.x = torch.cat(node_features_list, dim=0)
        data.edge_index = torch.cat(edge_index_list, dim=1)
        data.edge_attr = torch.cat(edge_features_list, dim=0)
        data.y = torch.cat(label_list, dim=0)

        torch.save(data, os.path.join(self.processed_dir, "data.pt"))

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return len(self.data.x)

    def get(self):
        data_file = os.path.join(self.processed_dir, "data.pt")
        if not os.path.exists(data_file):
            self.process()
        return torch.load(data_file)
